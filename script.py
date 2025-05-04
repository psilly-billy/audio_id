import os
import subprocess
import argparse
import glob
import time
import json
import requests
import torch
import librosa
import numpy as np
import uuid
import shutil
import sys
from collections import Counter
from functools import lru_cache

# Check for required system dependencies
def check_dependencies():
    """Check if required system dependencies are installed"""
    try:
        # Check for ffmpeg
        ffmpeg_version = subprocess.run(["ffmpeg", "-version"], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE, 
                                       text=True, 
                                       check=False)
        if ffmpeg_version.returncode != 0:
            print("WARNING: ffmpeg not found. Audio extraction will fail.")
            return False
        
        # Get first line of output without using backslash in f-string
        first_line = ffmpeg_version.stdout.split('\n')[0] if ffmpeg_version.stdout else "Unknown version"
        print(f"Found ffmpeg: {first_line}")
        return True
    except Exception as e:
        print(f"WARNING: Error checking dependencies: {e}")
        return False

# For handling environment variables and config
def get_hf_token():
    """Get Hugging Face token from environment or config file"""
    # Try to get from environment variable first (Streamlit Secrets or .env)
    hf_token = os.environ.get("HF_TOKEN")
    
    # If not in environment, try to import from config.py (for local development)
    if not hf_token:
        try:
            import config
            hf_token = config.HF_TOKEN
        except (ImportError, AttributeError):
            pass
    
    return hf_token

# Create a unique session ID for each analysis run
def generate_session_id():
    """Generate a unique session ID for this analysis run"""
    return f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"

# Video Downloading Module using yt-dlp
def download_video(video_url, output_dir="downloads", session_id=None):
    """
    Download video with a unique filename
    
    Args:
        video_url: URL of the video to download
        output_dir: Directory to save the video
        session_id: Optional session ID for creating unique filenames
    
    Returns:
        Path to the downloaded video file
    """
    if not session_id:
        session_id = generate_session_id()
        
    # Create session directory
    session_dir = os.path.join(output_dir, session_id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    # Use a unique output template with session ID
    output_template = os.path.join(session_dir, "video")
    command = ["yt-dlp", "-o", output_template, video_url]
    
    # Run the download process
    subprocess.run(command, check=True)
    
    # Find the actual downloaded file (could be .mp4, .webm, etc.)
    downloaded_files = glob.glob(f"{output_template}.*")
    if not downloaded_files:
        raise FileNotFoundError(f"No video file found in {session_dir} after download")
    
    # Return the path to the downloaded file
    return downloaded_files[0], session_id

# Extract audio from video with better error handling
def extract_audio(video_path, audio_format="wav", output_dir="audio", duration=None, session_id=None):
    """
    Extract audio from video with optional duration limit
    
    Args:
        video_path: Path to the video file
        audio_format: Format to save the audio (default: wav)
        output_dir: Directory to save the audio
        duration: Maximum duration in seconds to extract (None for full audio)
        session_id: Session ID for creating unique filenames
    
    Returns:
        Path to the extracted audio file
    """
    if not session_id:
        session_id = generate_session_id()
        
    # Create session directory
    session_dir = os.path.join(output_dir, session_id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
        
    audio_path = os.path.join(session_dir, "audio.wav")
    
    print(f"Extracting audio from {video_path} to {audio_path}")
    
    # First check if ffmpeg is available
    try:
        # Test ffmpeg
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        error_msg = f"ERROR: ffmpeg is not installed or not found in PATH. Audio extraction failed: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # Basic command
    command = ["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000"]
    
    # Add duration limit if specified
    if duration:
        print(f"Limiting audio to first {duration} seconds")
        command.extend(["-t", str(duration)])
    
    # Add output path
    command.append(audio_path)
    
    try:
        subprocess.run(command, check=True)
        return audio_path, session_id
    except subprocess.SubprocessError as e:
        error_msg = f"Error extracting audio: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)

# Check if GPU is available
def is_gpu_available():
    return torch.cuda.is_available()

# Advanced audio segmentation - single speaker mode
def segment_audio(audio_path, segment_duration=10, min_duration=1.0):
    """Split audio into equal segments assuming single speaker"""
    import librosa
    
    print(f"Segmenting audio file: {audio_path}")
    duration = librosa.get_duration(path=audio_path)
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Create segments by splitting audio into equal parts
    segments = []
    
    for start in range(0, int(duration), int(segment_duration)):
        end = min(start + segment_duration, duration)
        
        # Skip segments that are too short
        if end - start < min_duration:
            continue
            
        # Single speaker mode - all segments are from the same speaker
        speaker = "SPEAKER_0"
        segments.append({
            "speaker": speaker,
            "start": start,
            "end": end
        })
    
    print(f"Created {len(segments)} segments with 1 speaker")
    return segments

# Extract audio segments
def extract_segments(audio_path, segments, output_dir="audio_segments", session_id=None):
    """Extract audio segments to individual files"""
    if not session_id:
        session_id = generate_session_id()
        
    # Create session directory for segments
    session_dir = os.path.join(output_dir, session_id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    segment_files = []
    for segment in segments:
        start, end = segment["start"], segment["end"]
        speaker = segment["speaker"]
        
        # Create output filename
        segment_filename = f"segment_{speaker}_{start:.1f}_{end:.1f}.wav"
        segment_path = os.path.join(session_dir, segment_filename)
        
        print(f"Extracting segment from {start:.1f}s to {end:.1f}s")
        
        # Extract segment using ffmpeg
        try:
            subprocess.run([
                "ffmpeg", "-i", audio_path, 
                "-ss", str(start), "-to", str(end), 
                "-ac", "1", "-ar", "16000",
                "-y", segment_path  # -y to overwrite
            ], check=True)
            
            segment_files.append({
                "segment": segment,
                "path": segment_path
            })
        except Exception as e:
            print(f"Error extracting segment {start:.1f}-{end:.1f}: {e}")
    
    return segment_files, session_id

# Load Distil-Whisper model (cached to avoid reloading)
@lru_cache(maxsize=1)
def load_speech_model():
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        # Get HF token
        hf_token = get_hf_token()
        
        # Select device based on availability
        device = "cuda:0" if is_gpu_available() else "cpu"
        print(f"Device set to use {device}")
        
        # Use Distil-Whisper model - more efficient than Whisper Large
        model_id = "distil-whisper/distil-large-v3.5"
        
        # Handle torch CPU/GPU quirks
        torch_dtype = None
        if torch.cuda.is_available():
            torch_dtype = torch.float16
            print("Using float16 precision with GPU")
        else:
            # Use default dtype for CPU
            print("Using default precision with CPU")
        
        # Load the model and processor with auth token if available
        kwargs = {
            "low_cpu_mem_usage": True
        }
        
        # Only add torch_dtype if it's set (avoids errors on some platforms)
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
            
        # Add token if available
        if hf_token:
            kwargs["token"] = hf_token
            processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        else:
            # Attempt to load without token (for public models)
            processor = AutoProcessor.from_pretrained(model_id)
            
        # Load model with appropriate kwargs
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **kwargs)
        
        # Move model to the appropriate device
        model = model.to(device)
        
        return model, processor, device
    except Exception as e:
        print(f"Error loading Distil-Whisper model: {e}")
        raise

# Advanced accent detection using linguistic features
def detect_accent_advanced(transcription):
    """
    Advanced accent detection based on linguistic patterns, phonetics, and vocabulary
    
    Returns a tuple of (accent, confidence percentage)
    """
    # Convert to lowercase for analysis
    text = transcription.lower()
    
    # Define accent feature dictionaries with linguistic markers
    accent_features = {
        "Italian": {
            "phonetic_patterns": ["ciao", "buon", "sono", "molto", "questo", "perché", "è", "va", "bene"],
            "grammar_patterns": ["the is", "i have years", "we go to make", "i no like", "for make the"],
            "vocabulary": ["pizza", "pasta", "mamma", "casa", "bellissimo", "maestro", "ragazzi"]
        },
        "Spanish": {
            "phonetic_patterns": ["tch", "jou", "theenk", "ees", "de", "beecause", "espik"],
            "grammar_patterns": ["i no understand", "for to make", "i have years", "is very difficult for me"],
            "vocabulary": ["gracias", "hola", "amigo", "por favor", "bueno", "señor", "casa"]
        },
        "French": {
            "phonetic_patterns": ["ze", "zis", "sink", "sank", "ze", "how you say", "ow do you say"],
            "grammar_patterns": ["i am agree", "i am here since 3 years", "i have 30 years"],
            "vocabulary": ["voilà", "bonjour", "monsieur", "madame", "très", "beaucoup", "merci"]
        },
        "German": {
            "phonetic_patterns": ["ve", "vill", "zat", "zis", "vat", "ze", "sh", "ja"],
            "grammar_patterns": ["i become hungry", "i bekam", "it gives", "i make sport", "since long time"],
            "vocabulary": ["ja", "nein", "gut", "danke", "bitte", "schön", "naturlich"]
        },
        "Russian": {
            "phonetic_patterns": ["ze", "is", "dat", "dis", "zey", "vat", "eez", "howf", "wiz"],
            "grammar_patterns": ["i no have", "i not understand", "you want tea", "we go shop", "i from russia"],
            "vocabulary": ["da", "niet", "babushka", "tovarish", "spasiba", "vodka", "dobry"]
        },
        "Indian": {
            "phonetic_patterns": ["deh", "da", "tee", "ji", "yaar", "dem", "dose", "w replacing v"],
            "grammar_patterns": ["i am knowing", "i am having", "i was going", "we are wanting", "itself", "only"],
            "vocabulary": ["actually", "kindly", "do the needful", "prepone", "updation", "revert", "ji"]
        },
        "Chinese": {
            "phonetic_patterns": ["r/l confusion", "no final consonants", "sh for s", "l for r", "f for th", "no plural s"],
            "grammar_patterns": ["no articles", "no verb tense", "measure words", "no plurals", "topic prominence"],
            "vocabulary": ["um", "ah", "hao", "ma", "hai", "ya", "okay la", "actually", "so"]
        },
        "American": {
            "phonetic_patterns": ["r pronounced", "t flapping", "o as ah", "dropped t", "glottal stop"],
            "grammar_patterns": ["gotten", "I just ate", "I've already eaten", "on the weekend", "aside from"],
            "vocabulary": ["gonna", "wanna", "y'all", "awesome", "buddy", "fall", "truck", "apartment"]
        },
        "British": {
            "phonetic_patterns": ["non-rhotic r", "t pronounced", "round o", "pronounced h"],
            "grammar_patterns": ["I've got", "at the weekend", "in hospital", "at university", "have got"],
            "vocabulary": ["brilliant", "cheers", "mate", "bloody", "proper", "quite", "rather", "flat"]
        }
    }
    
    # Count matches for each accent
    scores = {}
    total_features = 0
    
    for accent, features in accent_features.items():
        score = 0
        feature_count = 0
        
        # Check for phonetic patterns
        for pattern in features["phonetic_patterns"]:
            if pattern in text or (len(pattern.split()) > 1 and any(p in text for p in pattern.split('/'))):
                score += 1
        feature_count += len(features["phonetic_patterns"])
        
        # Check for grammar patterns
        for pattern in features["grammar_patterns"]:
            if pattern in text:
                score += 2  # Grammar is a stronger indicator
        feature_count += len(features["grammar_patterns"]) * 2
        
        # Check for vocabulary
        for word in features["vocabulary"]:
            if word in text.split() or f" {word} " in f" {text} ":
                score += 1.5  # Vocabulary is a medium-strength indicator
        feature_count += len(features["vocabulary"]) * 1.5
        
        # Calculate weighted score and add to scores
        scores[accent] = score
        total_features += feature_count
    
    # If no significant matches, check additional context clues
    if sum(scores.values()) == 0:
        if "pizza" in text or "italian" in text or "italia" in text:
            scores["Italian"] = 1
        elif "spanish" in text or "españa" in text:
            scores["Spanish"] = 1
        # Add more context clues as needed
    
    # Determine most likely accent
    if not scores or sum(scores.values()) == 0:
        return "Unknown", 30.0  # Default confidence
    
    most_likely = max(scores.items(), key=lambda x: x[1])
    accent = most_likely[0]
    
    # Calculate confidence
    if sum(scores.values()) > 0:
        # Normalize confidence score to a percentage (max 95)
        confidence = min(95, (most_likely[1] / max(1, sum(scores.values()))) * 100)
        
        # Bump up confidence for very strong matches
        if most_likely[1] > 5:
            confidence = min(95, confidence + 10)
    else:
        confidence = 30.0  # Default confidence
    
    return accent, confidence

# Accent detection using Distil-Whisper and advanced linguistic analysis
def detect_accent(segment_file):
    """Detect accent in audio segment using Distil-Whisper and linguistic analysis"""
    try:
        # Load model
        model, processor, device = load_speech_model()
        
        # Load audio data
        waveform, sample_rate = librosa.load(segment_file, sr=16000)
        
        # Process the audio with the processor
        input_features = processor(
            waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate tokens 
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode the tokens to get the transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"Transcription: {transcription}")
        
        # Advanced accent detection based on transcription
        accent, confidence = detect_accent_advanced(transcription)
        
        return accent, confidence, transcription
    except Exception as e:
        print(f"Error in accent detection: {e}")
        return "Unknown", 0.0, ""

# Process all segments with aggregation
def analyze_segments(segment_files):
    """Process all segments and detect accents with aggregation across segments"""
    segment_results = []
    
    for file_info in segment_files:
        segment = file_info["segment"]
        path = file_info["path"]
        
        print(f"Analyzing accent for speaker {segment['speaker']} ({segment['start']:.1f}s - {segment['end']:.1f}s)")
        
        try:
            accent, confidence, transcription = detect_accent(path)
            
            segment_results.append({
            "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["end"] - segment["start"],
            "accent": accent,
                "confidence": confidence,
                "transcription": transcription
            })
            
            print(f"  → Detected accent: {accent} (confidence: {confidence:.1f}%)")
            print(f"  → Transcription: {transcription}")
        except Exception as e:
            print(f"Error processing segment: {e}")
    
    # Now aggregate results across all segments for better accuracy
    aggregate_results = aggregate_speaker_results(segment_results)
    
    # Merge the aggregated results back into the segment results
    for segment in segment_results:
        speaker = segment["speaker"]
        if speaker in aggregate_results:
            # Keep original transcription but update accent from aggregation
            segment["accent"] = aggregate_results[speaker]["accent"]
            segment["confidence"] = aggregate_results[speaker]["confidence"]
    
    return segment_results

# Aggregate results across segments for better accuracy
def aggregate_speaker_results(segment_results):
    """Aggregate results across all segments for each speaker"""
    # Group by speaker
    speakers = {}
    for result in segment_results:
        speaker = result["speaker"]
        if speaker not in speakers:
            speakers[speaker] = {
                "segments": [],
                "accents": [],
                "confidences": [],
                "durations": [],
                "transcriptions": []
            }
        
        speakers[speaker]["segments"].append(result)
        speakers[speaker]["accents"].append(result["accent"])
        speakers[speaker]["confidences"].append(result["confidence"])
        speakers[speaker]["durations"].append(result["duration"])
        speakers[speaker]["transcriptions"].append(result["transcription"])
    
    # Aggregate for each speaker
    aggregated = {}
    for speaker, data in speakers.items():
        # Weighted voting for accent based on duration and confidence
        accent_weights = {}
        for i, accent in enumerate(data["accents"]):
            if accent == "Unknown":
                continue
                
            weight = data["durations"][i] * data["confidences"][i]
            if accent not in accent_weights:
                accent_weights[accent] = 0
            accent_weights[accent] += weight
        
        # Determine most likely accent
        if accent_weights:
            most_likely = max(accent_weights.items(), key=lambda x: x[1])
            final_accent = most_likely[0]
            
            # Calculate aggregated confidence
            total_weight = sum(accent_weights.values())
            if total_weight > 0:
                final_confidence = min(95, (most_likely[1] / total_weight) * 100)
            else:
                final_confidence = 50.0  # Default for aggregated result
        else:
            final_accent = "Unknown"
            final_confidence = 30.0
        
        # Store the aggregate result
        aggregated[speaker] = {
            "accent": final_accent,
            "confidence": final_confidence,
            "total_duration": sum(data["durations"])
        }
    
    return aggregated

# Clean up temporary files
def cleanup_files(session_id=None, keep_results=False):
    """
    Clean up temporary files created during analysis
    
    Args:
        session_id: Session ID to clean up specific files
        keep_results: Whether to keep results files (default: False)
    """
    try:
        # Directories to clean
        dirs_to_clean = []
        
        if session_id:
            # Clean specific session directories
            dirs_to_clean = [
                os.path.join("downloads", session_id),
                os.path.join("audio", session_id),
                os.path.join("audio_segments", session_id)
            ]
        else:
            # Clean all directories (only if keep_results is False)
            if not keep_results:
                dirs_to_clean = ["downloads", "audio", "audio_segments"]
        
        # Remove directories
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                print(f"Cleaning up temporary files in {dir_path}")
                shutil.rmtree(dir_path)
                
        print("Cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Main workflow with dependency check
def run_accent_analysis(video_url, sample_duration=60, cleanup=True):
    """Run the full accent analysis pipeline"""
    session_id = generate_session_id()
    print(f"Starting analysis session: {session_id}")
    
    # Check dependencies first
    check_dependencies()
    
    try:
        print(f"Downloading video from: {video_url}")
        video_file, session_id = download_video(video_url, session_id=session_id)
        
        print(f"Extracting audio from video...")
        if sample_duration:
            print(f"Using only the first {sample_duration} seconds for analysis")
        audio_file, session_id = extract_audio(video_file, duration=sample_duration, session_id=session_id)
        
        print(f"Segmenting audio (single speaker mode)...")
        segments = segment_audio(audio_file, segment_duration=10)  # 10-second segments
        
        print(f"Extracting {len(segments)} segments...")
        segment_files, session_id = extract_segments(audio_file, segments, session_id=session_id)
        
        print(f"Analyzing accents for {len(segment_files)} audio segments...")
        results = analyze_segments(segment_files)
        
        # Clean up temporary files if requested
        if cleanup:
            cleanup_files(session_id)
            
        return results
    except Exception as e:
        print(f"Error in analysis: {e}")
        # Always try to clean up on error
        if cleanup:
            cleanup_files(session_id)
        raise

# Main function
if __name__ == "__main__":
    # Set up environment variables if .env file exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv is optional
    
    # Check dependencies
    check_dependencies()
    
    parser = argparse.ArgumentParser(description="Analyze accents in a video")
    parser.add_argument("--video_url", type=str, help="URL of the video to analyze")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds to analyze (default: 60s)")
    parser.add_argument("--full", action="store_true", help="Process the full video instead of a sample")
    parser.add_argument("--keep_files", action="store_true", help="Don't delete temporary files after processing")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token (if not set in environment variables)")
    args = parser.parse_args()
    
    # Set HF token from command line if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    # Use command-line URL or default
    VIDEO_URL = args.video_url if args.video_url else "https://example.com/path_to_video.mp4"
    if VIDEO_URL == "https://example.com/path_to_video.mp4":
        print("Warning: Using example URL. Please provide a real video URL with --video_url")
    
    # Set duration (None if --full is specified)
    DURATION = None if args.full else args.duration
    
    # Set cleanup flag (False if --keep_files is specified)
    CLEANUP = not args.keep_files
    
    results = run_accent_analysis(VIDEO_URL, sample_duration=DURATION, cleanup=CLEANUP)
    print("\n===== RESULTS =====")
    for res in results:
        print(f"Speaker {res['speaker']} (from {res['start']:.1f} to {res['end']:.1f} sec):")
        print(f"  Accent Classification: {res['accent']} with {res['confidence']:.1f}% confidence")
        print(f"  Transcription: {res['transcription']}")