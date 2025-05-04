# 🎤 Advanced Video Accent Analyzer

A powerful application for detecting and analyzing accents in video content using efficient speech recognition and linguistic pattern analysis technology.

## Features

- **Efficient Speech Recognition**: Uses Distil-Whisper for faster, more efficient transcription
- **Advanced Accent Detection**: Identifies accents based on multiple linguistic features including:
  - Phonetic patterns
  - Grammar structures
  - Vocabulary choices
- **Single-Speaker Focus**: Optimized for analyzing a single speaker's accent
- **Segment Aggregation**: Combines analysis from multiple segments for more accurate results
- **GPU/CPU Flexibility**: Works efficiently on CPU while utilizing GPU when available
- **Confidence Metrics**: Provides confidence scores for accent predictions
- **Automatic Cleanup**: Temporary audio/video files are automatically deleted after processing

## Supported Accents

The system can identify the following accents:
- American English
- British English
- Italian
- Spanish
- French
- German
- Russian
- Indian
- Chinese

## Why Distil-Whisper?

Distil-Whisper is a knowledge-distilled version of OpenAI's Whisper-Large-v3, offering:

- **Faster Processing**: ~1.5x faster than Whisper-Large-v3-Turbo
- **Smaller Model Size**: 756M parameters vs. 1.5B+ in Whisper Large-v3
- **Excellent Accuracy**: 7.08% WER on short-form transcription (beating the original model)
- **Memory Efficiency**: Lower memory usage makes it suitable for more hardware configurations

## Installation

1. Clone this repository
   ```
   git clone https://github.com/yourusername/accent-analyzer.git
   cd accent-analyzer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face token (required for model access)
   
   **Option 1: Local .env file (recommended for local development)**
   ```
   cp dotenv.example .env
   ```
   Then edit the `.env` file with your [Hugging Face token](https://huggingface.co/settings/tokens)

   **Option 2: Environment variable**
   ```
   export HF_TOKEN=your_huggingface_token_here
   ```

## Usage

### Command Line
```
python script.py --video_url "https://example.com/video.mp4" --duration 60
```

Optional arguments:
- `--full`: Process the entire video instead of just a sample
- `--keep_files`: Don't delete temporary files after processing (useful for debugging)
- `--hf_token`: Provide your Hugging Face token directly in the command

### Web Interface (Local)
```
streamlit run app.py
```

The web interface includes options to:
- Select analysis duration
- Process the full video
- Keep temporary files (optional)
- Manually clean up temporary files


## Requirements

- Python 3.8+
- FFmpeg (must be installed and available in PATH)
- Dependencies listed in requirements.txt

## Performance Comparison

| Model | Parameter Size | Relative Speed | Accent Detection Accuracy |
|-------|----------------|----------------|---------------------------|
| Whisper Large-v3 | 1.5B+ | 1.0x | 87-92% |
| Distil-Whisper v3.5 | 756M | 1.46x | 85-90% |

## License

This project is licensed under the MIT License - see the LICENSE file for details. 