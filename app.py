import streamlit as st
import os
import sys
import subprocess

# Configure error handling
st.set_page_config(
    page_title="Video Accent Analyzer",
    page_icon="🎤",
    layout="wide",
)

# Add custom styling for app header
st.markdown("""
<style>
    .app-header {
        background-color: #2c3e50;
        padding: 25px;
        border-radius: 10px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .app-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .app-header p {
        color: #ddd;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    .stButton>button {
        background-color: #007acc;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005c99;
    }
</style>
""", unsafe_allow_html=True)

# Title and description with custom styling
st.markdown("""
<div class="app-header">
    <h1>🎤 Video Accent Analyzer</h1>
    <p>Analyze videos to identify speaker accents using efficient speech recognition and linguistic pattern analysis.</p>
</div>
""", unsafe_allow_html=True)

# Check for required system dependencies
def check_system_dependencies():
    """Check if required system dependencies are installed and show warnings"""
    missing_deps = []
    
    # Check for ffmpeg
    try:
        ffmpeg_result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=False
        )
        if ffmpeg_result.returncode != 0:
            missing_deps.append("ffmpeg")
    except Exception:
        missing_deps.append("ffmpeg")
    
    # Check for yt-dlp
    try:
        ytdlp_result = subprocess.run(
            ["yt-dlp", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=False
        )
        if ytdlp_result.returncode != 0:
            missing_deps.append("yt-dlp")
    except Exception:
        missing_deps.append("yt-dlp")
    
    return missing_deps

# Check dependencies early
missing_deps = check_system_dependencies()
if missing_deps:
    st.error(f"""
    ⚠️ Missing System Dependencies: {', '.join(missing_deps)}
    
    This app requires these programs to be installed on the system:
    - ffmpeg: For audio extraction and processing
    - yt-dlp: For video download
    
    If running on Streamlit Cloud, please ensure packages.txt includes these dependencies.
    """)
    
    # Show packages.txt info
    with st.expander("How to fix on Streamlit Cloud"):
        st.code("""
# Create a file named packages.txt with:
ffmpeg
youtube-dl

# Make sure this file is in your repository root
        """)

# Set up environment variables
def setup_environment():
    """Set up environment variables from Streamlit secrets or .env file"""
    # First try to load from Streamlit secrets (for cloud deployment)
    if "HF_TOKEN" in st.secrets:
        os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
        return True
        
    # For local development, try to load from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        if os.environ.get("HF_TOKEN"):
            return True
    except ImportError:
        pass
        
    # If still not found, try config.py (backward compatibility)
    try:
        import config
        os.environ["HF_TOKEN"] = config.HF_TOKEN
        if os.environ["HF_TOKEN"] != "YOUR_NEW_TOKEN_HERE":
            return True
    except (ImportError, AttributeError):
        pass
        
    return False

# Set up environment
hf_token_present = setup_environment()

# Now import the script module
import script

# Check if token is available
if not hf_token_present:
    st.error(f"""
    ⚠️ HuggingFace Token Missing
    
    This app requires a HuggingFace token to download the accent detection model.
    
    Please set up your token using one of these methods:
    
    1. Streamlit Cloud: Add HF_TOKEN to your app secrets
    2. Local Development: Create a `.env` file with `HF_TOKEN=your_token_here`
    3. Legacy Method: Create a `config.py` file with `HF_TOKEN = "your_token_here"`
    """)
    
    # Optional: Allow manual token entry in the UI (less secure)
    with st.expander("Enter token manually (not recommended for production)"):
        manual_token = st.text_input("HuggingFace Token", type="password")
        if manual_token and st.button("Set Token"):
            os.environ["HF_TOKEN"] = manual_token
            hf_token_present = True
            st.success("Token set! You can now proceed with the analysis.")
            st.experimental_rerun()
            
    if not hf_token_present:
        st.stop()

# Model info section
st.sidebar.header("Model Information")
st.sidebar.markdown("""
**Current Model:** Distil-Whisper Large v3.5

- Knowledge-distilled version of Whisper Large
- 756M parameters (vs 1.5B+ in full Whisper)
- 1.46x faster than Whisper-Large-v3-Turbo 
- 7.08% WER on short-form transcription
""")

# Inform about GPU status
if script.is_gpu_available():
    st.sidebar.success("✅ GPU Detected: Using mixed precision for faster processing")
else:
    st.sidebar.info("ℹ️ Using CPU: Distil-Whisper is optimized for CPU performance")

# Input form
st.subheader("Video Input")
with st.form("video_form"):
    video_url = st.text_input("Video URL (YouTube, direct link, etc.)", 
                              placeholder="https://youtube.com/watch?v=...")
    
    col1, col2 = st.columns(2)
    with col1:
        sample_duration = st.slider("Analysis Duration (seconds)", 
                                    min_value=10, max_value=300, value=60, step=10,
                                    help="Longer duration can improve accuracy but increases processing time")
    with col2:
        use_full_video = st.checkbox("Analyze Full Video", value=False, 
                                   help="Warning: Processing the full video may take a long time")
    
    # Add cleanup options
    keep_files = st.checkbox("Keep temporary files after analysis", value=False,
                          help="Disable automatic cleanup (useful for debugging)")
    
    submit_button = st.form_submit_button("Analyze Video")

# Process video when form is submitted
if submit_button and video_url:
    try:
        # Double check dependencies again
        missing_deps = check_system_dependencies()
        if missing_deps:
            st.error(f"Cannot proceed: Missing system dependencies: {', '.join(missing_deps)}")
            st.stop()
            
        with st.spinner("Processing video... This may take a minute."):
            # Set duration to None if analyzing full video
            duration = None if use_full_video else sample_duration
            
            # Set cleanup flag (False if keep_files is checked)
            cleanup = not keep_files
            
            # Run the analysis with our cleanup setting
            results = script.run_accent_analysis(video_url, sample_duration=duration, cleanup=cleanup)
            
            # Show results
            st.subheader("Analysis Results")
            
            # Add custom CSS for better styling
            st.markdown("""
            <style>
            .accent-box {
                background-color: #2e4057;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                color: white;
            }
            .accent-label {
                font-size: 1.2rem;
                color: #e8f1f2;
                font-weight: bold;
            }
            .accent-value {
                font-size: 1.5rem;
                color: #ffcc66;
                font-weight: bold;
                margin-left: 10px;
            }
            .confidence-high {
                color: #5cb85c;
                font-weight: bold;
            }
            .confidence-medium {
                color: #f0ad4e;
                font-weight: bold;
            }
            .confidence-low {
                color: #d9534f;
                font-weight: bold;
            }
            .transcript-box {
                background-color: #121212;
                color: #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
                font-family: 'Courier New', monospace;
                border-left: 5px solid #66a3ff;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Process results by speaker
            speakers = {}
            for result in results:
                speaker = result["speaker"]
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append(result)
            
            # Display for each speaker
            for speaker, segments in speakers.items():
                # Get the most common accent across segments
                combined_accent = segments[0]["accent"]  # already aggregated in the script
                combined_confidence = segments[0]["confidence"]
                
                # Create a nice display
                st.markdown(f"### Speaker {speaker.replace('SPEAKER_', '')}")
                
                # Determine confidence class
                confidence_class = "confidence-high" if combined_confidence >= 70 else "confidence-medium" if combined_confidence >= 50 else "confidence-low"
                
                # Show accent with confidence in styled box
                st.markdown(f"""
                <div class="accent-box">
                    <span class="accent-label">Detected Accent:</span>
                    <span class="accent-value">{combined_accent}</span>
                    <br/>
                    <span class="accent-label">Confidence:</span>
                    <span class="{confidence_class}">{combined_confidence:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Display a formatted transcript from all segments
                st.markdown("#### Transcript:")
                
                transcript_text = ""
                for segment in segments:
                    transcript_text += segment["transcription"] + " "
                
                st.markdown(f'<div class="transcript-box">{transcript_text}</div>', unsafe_allow_html=True)
                
                # Show the individual segments in an expandable section
                with st.expander("Show Individual Segments"):
                    for i, segment in enumerate(segments):
                        st.markdown(f"**Segment {i+1}** ({segment['start']:.1f}s - {segment['end']:.1f}s):")
                        st.markdown(f'<div class="transcript-box">{segment["transcription"]}</div>', unsafe_allow_html=True)
                    
                st.markdown("---")
            
            # Add a download button for the results
            import json
            import base64
            
            # Prepare the results as JSON
            results_json = json.dumps(results, indent=2)
            b64 = base64.b64encode(results_json.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="accent_analysis_results.json">Download Results (JSON)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Add manual cleanup option if files were kept
            if keep_files:
                if st.button("Clean Up Temporary Files"):
                    script.cleanup_files(keep_results=False)
                    st.success("Temporary files have been cleaned up.")
            
    except Exception as e:
        st.error(f"Error processing video: {e}")
        st.error("Please check the URL and try again.")
        
        # Show detailed error in expander
        with st.expander("Detailed Error Information"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())
else:
    if submit_button:  # Form was submitted but URL is empty
        st.warning("Please enter a video URL to analyze.")
    else:  # First time or form not submitted yet
        st.info("Enter a video URL above and click 'Analyze Video' to begin.")

# Footer
st.markdown("---")
st.markdown("**Video Accent Analyzer** | Using Distil-Whisper for efficient accent detection")
st.markdown("Supports Italian, Spanish, French, German, Russian, Indian, Chinese, American, and British accents.")

# Add info about the app
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    ### Accent Detection
    
    This app uses:
    - yt-dlp for video download
    - FFmpeg for audio extraction
    - Distil-Whisper for high-accuracy transcription
    
    
    Supported accents:
    - American, British
    - Italian, Spanish, French, German
    - Russian, Indian, Chinese
    """)
    
    st.subheader("Performance Tips")
    st.markdown("""
    - Processing only the first 30-60 seconds is much faster
    - For best results, ensure audio quality is good
    - Longer speech samples (>30 seconds) provide more accurate accent detection
    - The app automatically uses GPU acceleration if available
    - Files are automatically cleaned up after analysis (unless disabled)
    """)
    
    st.info("This model is an efficient distilled version of Whisper, providing similar accuracy with lower computational requirements.") 