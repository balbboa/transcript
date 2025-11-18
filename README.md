# Anime Audio Transcriber

A specialized GUI tool optimized for transcribing **anime audio** into SRT subtitle files using OpenAI's Whisper. Specifically tuned for anime's unique audio characteristics.

## Requirements

- **Python 3.12** (or earlier)
- **FFmpeg** (for MP3, M4A, and other audio formats)

## Installation

Open **Windows PowerShell** and run the following commands:

### 1. Install Python 3.12

Download from [python.org](https://www.python.org/downloads/) if needed.

### 2. Install FFmpeg

```powershell
winget install ffmpeg
```

Or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### 3. Automatic Setup (Recommended)

Run the automatic setup script that detects your GPU and installs everything:

```powershell
python setup_gpu.py
```

This will automatically:
- ✅ Detect your NVIDIA GPU (if available)
- ✅ Detect your CUDA version
- ✅ Install the correct PyTorch version with GPU support
- ✅ Install all other dependencies
- ✅ Verify everything works

**If you have a GPU**, this will enable 3-5x faster transcription!

### 3b. Manual Installation (Alternative)

If you prefer manual installation:

```powershell
pip install -r requirements.txt
```

Then install PyTorch with CUDA support based on your CUDA version (see [PyTorch website](https://pytorch.org/get-started/locally/)).

## Usage

### Run the Application

Open **Windows PowerShell** and run:

```powershell
python jp_transcriber.py
```

### How to Use

1. Select a Whisper model (tiny/base/small/medium/large/large-v2/large-v3)
   - **Recommended: large-v3** for maximum accuracy (default)
   - Smaller models are faster but less accurate
2. Choose an audio file or drag & drop it
3. Click "Start Transcription"
4. Find the output files in the same folder as your audio:
   - `[filename]_jp.txt` - Text file with timestamps
   - `[filename]_jp.srt` - SRT subtitle file

## Notes

- First run will download the Whisper model (~1.4 GB for medium, ~3 GB for large-v3)
- Supported formats: MP3, WAV, M4A, AAC, FLAC, OGG
- **Optimized for Anime Audio**: Speed-optimized while maintaining accuracy for anime's unique challenges:
  - **Fast speech**: Optimized beam search (5 candidates) and focused temperature sampling
  - **Overlapping voices**: More lenient confidence thresholds to capture all dialogue
  - **Slang & casual speech**: Anime-specific context prompts for better recognition
  - **Shouting & whispering**: Lower VAD threshold (0.3) to catch quiet moments and loud exclamations
  - **Background music & SFX**: Higher compression threshold (3.0) to filter repetitive BGM patterns
  - **Anime expressions**: Enhanced normalization for elongated sounds (えー, あー, etc.) and exaggerated punctuation
  - Beam search with 5 candidates (optimized for speed)
  - Temperature sampling with 4 key values (0.0, 0.2, 0.4, 0.6) for faster processing
  - Voice activity detection optimized for anime audio
  - Japanese-specific punctuation and particle handling
  - Context-aware transcription with anime-specific prompts
  - **~30-40% faster** than maximum accuracy settings while maintaining excellent quality
