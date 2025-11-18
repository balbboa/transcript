import os
import re
import threading
import time
import unicodedata
import warnings
from collections import deque
from datetime import datetime, timedelta
from tkinter import Tk, Label, Button, filedialog, Text, END, ttk, StringVar
from tkinterdnd2 import DND_FILES, TkinterDnD

# Suppress expected warnings about Triton/CUDA (fallback to CPU is fine)
warnings.filterwarnings("ignore", message=".*Triton kernels.*")
warnings.filterwarnings("ignore", message=".*CUDA toolkit.*")
warnings.filterwarnings("ignore", message=".*falling back to.*")

import whisper

# Try to import mutagen for audio duration, fallback to ffprobe if available
try:
    from mutagen import File as MutagenFile
    from mutagen.mp3 import MP3
    from mutagen.wave import WAVE
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

# Note: Voice Activity Detection (VAD) is handled by Whisper internally
# through the no_speech_threshold parameter and its built-in silence detection

# ======================================================
# ----------- Core Transcription Utilities -------------
# ======================================================

def get_audio_duration(audio_path):
    """Get audio file duration in seconds using mutagen or fallback methods"""
    try:
        if HAS_MUTAGEN:
            audio_file = MutagenFile(audio_path)
            if audio_file is not None and hasattr(audio_file, 'info'):
                duration = audio_file.info.length
                if duration and duration > 0:
                    return duration
    except Exception:
        pass
    
    # Fallback: try using ffprobe if available
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            if duration > 0:
                return duration
    except Exception:
        pass
    
    # Last resort: estimate based on file size (very rough)
    try:
        file_size = os.path.getsize(audio_path)
        # Rough estimate: ~1MB per minute for MP3
        estimated_duration = (file_size / (1024 * 1024)) * 60
        return max(estimated_duration, 10)  # Minimum 10 seconds
    except Exception:
        return 60  # Default fallback: 1 minute

# ======================================================
# ----------- ETA Tracker with Advanced Features -------
# ======================================================

class ETATracker:
    """
    Advanced ETA tracker with:
    - Real work units (audio seconds)
    - Rolling average speed
    - Exponential moving average smoothing
    - Stabilization phase
    - Weighted stages
    - Continuous recalculation
    - Smooth ETA adjustments
    - Indeterminate mode for high uncertainty
    """
    
    def __init__(self, total_work_units, stage_weights=None, stabilization_samples=5, 
                 smoothing_alpha=0.3, rolling_window_size=10, uncertainty_threshold=0.5):
        """
        Args:
            total_work_units: Total work to complete (e.g., audio duration in seconds)
            stage_weights: Dict of stage_name -> weight (default: equal weights)
            stabilization_samples: Number of samples before showing ETA
            smoothing_alpha: EMA smoothing factor (0-1, lower = more smoothing)
            rolling_window_size: Size of rolling average window
            uncertainty_threshold: Coefficient of variation threshold for indeterminate mode
        """
        self.total_work_units = total_work_units
        self.stage_weights = stage_weights or {}
        self.stabilization_samples = stabilization_samples
        self.smoothing_alpha = smoothing_alpha
        self.rolling_window_size = rolling_window_size
        self.uncertainty_threshold = uncertainty_threshold
        
        # State tracking
        self.start_time = None
        self.last_update_time = None
        self.completed_work_units = 0.0
        self.current_stage = None
        self.stage_start_work = 0.0
        
        # Speed tracking
        self.speed_samples = deque(maxlen=rolling_window_size)  # Rolling window
        self.ema_speed = None  # Exponential moving average speed
        
        # Progress history for smoothing
        self.progress_history = deque(maxlen=20)
        self.eta_history = deque(maxlen=10)  # For ETA smoothing
        
        # Statistics
        self.sample_count = 0
        self.is_stabilized = False
        self.last_eta = None
        
    def start(self):
        """Start tracking"""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
    def update(self, completed_work_units, stage_name=None):
        """
        Update progress with completed work units
        
        Args:
            completed_work_units: Work units completed (e.g., seconds transcribed)
            stage_name: Current stage name (for weighted progress)
        """
        current_time = time.time()
        
        # Handle stage changes
        if stage_name and stage_name != self.current_stage:
            self.current_stage = stage_name
            self.stage_start_work = self.completed_work_units
        
        # Calculate work done since last update
        work_delta = completed_work_units - self.completed_work_units
        time_delta = current_time - self.last_update_time if self.last_update_time else 0.1
        
        # Update completed work
        self.completed_work_units = completed_work_units
        
        # Calculate instantaneous speed
        if time_delta > 0.01:  # Avoid division by very small numbers
            instant_speed = work_delta / time_delta
            
            # Add to rolling window
            self.speed_samples.append(instant_speed)
            
            # Update EMA speed
            if self.ema_speed is None:
                self.ema_speed = instant_speed
            else:
                self.ema_speed = (self.smoothing_alpha * instant_speed + 
                                 (1 - self.smoothing_alpha) * self.ema_speed)
        
        # Update timestamps
        self.last_update_time = current_time
        self.sample_count += 1
        
        # Check stabilization
        if not self.is_stabilized and self.sample_count >= self.stabilization_samples:
            self.is_stabilized = True
        
        # Store progress for smoothing
        self.progress_history.append({
            'work': completed_work_units,
            'time': current_time,
            'speed': self.ema_speed or 0
        })
    
    def get_progress_percent(self):
        """Get progress percentage (0-100)"""
        if self.total_work_units <= 0:
            return 0.0
        
        # Apply stage weights if available
        if self.current_stage and self.current_stage in self.stage_weights:
            # Calculate weighted progress within stage
            stage_weight = self.stage_weights[self.current_stage]
            stage_progress = min(1.0, self.completed_work_units / self.total_work_units)
            
            # Weight the progress by stage importance
            weighted_progress = stage_progress * stage_weight
            return min(100.0, weighted_progress * 100)
        
        return min(100.0, (self.completed_work_units / self.total_work_units) * 100)
    
    def get_eta_seconds(self):
        """
        Get ETA in seconds with smoothing and uncertainty handling
        
        Returns:
            ETA in seconds, or None if indeterminate
        """
        if not self.is_stabilized or self.completed_work_units >= self.total_work_units:
            return None
        
        if self.ema_speed is None or self.ema_speed <= 0:
            return None
        
        # Calculate remaining work
        remaining_work = self.total_work_units - self.completed_work_units
        
        # Calculate ETA using EMA speed
        eta = remaining_work / self.ema_speed
        
        # Check for high uncertainty (coefficient of variation)
        if len(self.speed_samples) >= 3:
            speeds = list(self.speed_samples)
            mean_speed = sum(speeds) / len(speeds)
            if mean_speed > 0:
                variance = sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)
                std_dev = variance ** 0.5
                cv = std_dev / mean_speed  # Coefficient of variation
                
                if cv > self.uncertainty_threshold:
                    return None  # Too uncertain, use indeterminate mode
        
        # Smooth ETA to prevent wild jumps
        if self.last_eta is not None:
            # Apply smoothing: new ETA is weighted average of old and new
            smoothing_factor = 0.4  # How much to trust new value
            eta = smoothing_factor * eta + (1 - smoothing_factor) * self.last_eta
        
        # Ensure ETA never goes to 0 until actually done
        if eta < 1.0 and remaining_work > 0:
            eta = 1.0  # Minimum 1 second
        
        self.last_eta = eta
        self.eta_history.append(eta)
        
        return eta
    
    def get_eta_string(self):
        """Get formatted ETA string"""
        eta_seconds = self.get_eta_seconds()
        
        if eta_seconds is None:
            return "Calculating..."
        
        # Format as human-readable time
        if eta_seconds < 60:
            return f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            minutes = int(eta_seconds // 60)
            seconds = int(eta_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_elapsed_time(self):
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_speed_string(self):
        """Get current speed as string (work units per second)"""
        if self.ema_speed is None or self.ema_speed <= 0:
            return "N/A"
        
        # For audio, show as "X seconds per second" or "Xx realtime"
        if self.ema_speed >= 1.0:
            return f"{self.ema_speed:.2f}x"
        else:
            return f"{1.0/self.ema_speed:.2f}s/s" if self.ema_speed > 0 else "N/A"
    
    def should_use_indeterminate(self):
        """Determine if progress bar should be indeterminate"""
        eta = self.get_eta_seconds()
        return eta is None or (not self.is_stabilized)
    
    def is_complete(self):
        """Check if work is complete"""
        return self.completed_work_units >= self.total_work_units

def log(text_widget, message, root=None):
    """Thread-safe logging function"""
    if root:
        root.after(0, lambda: _log_safe(text_widget, message))
    else:
        _log_safe(text_widget, message)

def _log_safe(text_widget, message):
    """Internal function to safely update text widget"""
    text_widget.insert(END, message + "\n")
    text_widget.see(END)

# ---------------- Text Normalization ------------------

def normalize_japanese(text):
    """Enhanced Japanese text normalization optimized for anime audio"""
    # Normalize Unicode (NFKC handles full-width/half-width, etc.)
    text = unicodedata.normalize("NFKC", text)
    
    # Remove excessive whitespace but preserve single spaces
    text = re.sub(r"\s+", " ", text)
    
    # Fix common spacing issues around Japanese punctuation
    # Japanese punctuation should not have spaces before or after
    text = re.sub(r"\s+([。、，！？：；])", r"\1", text)  # Remove space before punctuation
    text = re.sub(r"([。、，！？：；])\s+", r"\1", text)  # Remove space after punctuation
    
    # Fix spacing around quotation marks (Japanese uses 「」 and 『』)
    text = re.sub(r"\s+([「『])", r"\1", text)  # Remove space before opening quotes
    text = re.sub(r"([」』])\s+", r"\1", text)  # Remove space after closing quotes
    
    # Fix spacing around brackets and parentheses
    text = re.sub(r"\s+([（【])", r"\1", text)  # Remove space before opening brackets
    text = re.sub(r"([）】])\s+", r"\1", text)  # Remove space after closing brackets
    
    # Normalize repeated punctuation (anime often has exaggerated punctuation)
    # Keep some repetition for emphasis (common in anime), but normalize excessive repetition
    text = re.sub(r"([！？]){4,}", r"\1\1\1", text)  # Cap at 3 for emphasis
    text = re.sub(r"([。]){3,}", r"。", text)  # Multiple periods to single
    
    # Fix spacing around particles (助詞) - they should not have spaces before them
    # Common particles: は、が、を、に、で、と、から、まで、より、へ
    particles = r"[はがをにでとからまでよりへ]"
    text = re.sub(rf"\s+({particles})", r"\1", text)
    
    # Anime-specific: Normalize common anime expressions and interjections
    # These are often transcribed inconsistently
    text = re.sub(r"え[ー〜～]{2,}", "えー", text)  # Normalize elongated "e" sounds
    text = re.sub(r"あ[ー〜～]{2,}", "あー", text)  # Normalize elongated "a" sounds
    text = re.sub(r"う[ー〜～]{2,}", "うー", text)  # Normalize elongated "u" sounds
    text = re.sub(r"お[ー〜～]{2,}", "おー", text)  # Normalize elongated "o" sounds
    text = re.sub(r"い[ー〜～]{2,}", "いー", text)  # Normalize elongated "i" sounds
    
    # Remove zero-width spaces and other invisible characters
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    
    # Remove control characters but preserve newlines in multi-line text
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    
    # Normalize different types of dashes and hyphens to Japanese long vowel mark
    text = re.sub(r"[―－−‐]", "ー", text)
    
    # Normalize various tilde/wave characters used in anime for elongated sounds
    text = re.sub(r"[〜～]", "ー", text)  # Convert tildes to long vowel mark
    
    return text.strip()

def normalize_portuguese(text):
    """Enhanced Portuguese text normalization for better accuracy"""
    # Normalize Unicode (NFC for Portuguese)
    text = unicodedata.normalize("NFC", text)
    
    # Remove excessive whitespace but preserve single spaces
    text = re.sub(r"\s+", " ", text)
    
    # Fix common spacing issues around Portuguese punctuation
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)  # Remove space before punctuation
    text = re.sub(r"([.,!?:;])\s+", r"\1 ", text)  # Ensure single space after punctuation
    
    # Remove zero-width spaces and other invisible characters
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    
    # Fix common Portuguese punctuation issues
    text = re.sub(r"([a-zA-Z])\s+([,.:;!?])", r"\1\2", text)  # Remove space before punctuation after letters
    
    return text.strip()

# ----------------- Line Breaking ----------------------

def break_japanese_lines(text, max_chars=25):
    """Break Japanese text into lines at appropriate points"""
    lines = []
    current = ""

    for char in text:
        current += char
        if len(current) >= max_chars and char in "。、？！":
            lines.append(current.strip())
            current = ""

    if current:
        lines.append(current.strip())

    return lines

def break_portuguese_lines(text, max_chars=42):
    """Break Portuguese text into lines at appropriate points"""
    lines = []
    current = ""
    
    # Split by sentences first, then by words if needed
    sentences = re.split(r'([.!?]\s+)', text)
    
    for part in sentences:
        if len(current + part) <= max_chars:
            current += part
        else:
            if current:
                # Try to break at word boundaries
                words = current.split()
                line = ""
                for word in words:
                    if len(line + word) <= max_chars:
                        line += word + " "
                    else:
                        if line:
                            lines.append(line.strip())
                        line = word + " "
                if line:
                    lines.append(line.strip())
            current = part
    
    if current:
        words = current.split()
        line = ""
        for word in words:
            if len(line + word) <= max_chars:
                line += word + " "
            else:
                if line:
                    lines.append(line.strip())
                line = word + " "
        if line:
            lines.append(line.strip())
    
    return lines if lines else [text]

# ---------------- Timestamp Tools ---------------------

def sec_to_srt_time(seconds: float):
    td = timedelta(seconds=seconds)
    h, m, s = str(td).split(":")
    s = float(s)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"

# ------------------ SRT Creation ----------------------

def create_srt_from_segments(segments, output_path, log_widget=None, root=None):
    """
    Create SRT file from segments
    
    Args:
        segments: List of segment dictionaries with 'start', 'end', 'text'
        output_path: Output file path
        log_widget: Text widget for logging
        root: Tkinter root for thread-safe updates
    """
    if log_widget: log(log_widget, "→ Generating SRT...", root)

    # Use Japanese normalization and line breaking
    normalize_func = normalize_japanese
    break_lines_func = break_japanese_lines

    with open(output_path, "w", encoding="utf-8") as f:
        for index, seg in enumerate(segments, 1):
            start = sec_to_srt_time(seg["start"])
            end = sec_to_srt_time(seg["end"])

            text = normalize_func(seg["text"])
            lines = break_lines_func(text)

            f.write(f"{index}\n")
            f.write(f"{start} --> {end}\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")

    if log_widget: log(log_widget, f"✓ SRT saved: {output_path}", root)

# ---------------- Whisper Transcription/Translation ---------------

def transcribe_audio(audio_path, output_prefix="output", model_size="medium", 
                    log_widget=None, progress_callback=None, eta_tracker=None, root=None):
    """
    Transcribe Japanese anime audio with ETA tracking
    
    Args:
        audio_path: Path to audio file
        output_prefix: Prefix for output files
        model_size: Whisper model size
        log_widget: Text widget for logging
        progress_callback: Callback(progress_percent, eta_tracker, stage) for progress updates
        eta_tracker: ETATracker instance (will be created if None)
        root: Tkinter root for thread-safe updates
    """
    try:
        # Get audio duration for real work units
        audio_duration = get_audio_duration(audio_path)
        
        # Create ETA tracker if not provided
        if eta_tracker is None:
            # Define stage weights: model loading (5%), transcription (85%), file writing (10%)
            stage_weights = {
                'model_loading': 0.05,
                'transcription': 0.85,
                'file_writing': 0.10
            }
            eta_tracker = ETATracker(
                total_work_units=audio_duration,
                stage_weights=stage_weights,
                stabilization_samples=5,
                smoothing_alpha=0.3,
                rolling_window_size=10,
                uncertainty_threshold=0.5
            )
            eta_tracker.start()
        
        # Stage 1: Model Loading (5% of total work)
        if log_widget: 
            log(log_widget, f"→ Loading Whisper model '{model_size}'...", root)
        if progress_callback:
            progress_callback(0, eta_tracker, 'model_loading')
        
        model = whisper.load_model(model_size)
        
        # Check and log GPU availability with detailed diagnostics
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                if log_widget:
                    log(log_widget, f"✓ GPU detected: {gpu_name} (CUDA {cuda_version})", root)
                    log(log_widget, f"✓ GPU count: {gpu_count}", root)
            else:
                # Provide detailed diagnostics for why GPU is not available
                diagnostics = []
                
                # Check if PyTorch was built with CUDA support
                pytorch_cuda = getattr(torch.version, 'cuda', None)
                if not pytorch_cuda:
                    diagnostics.append("PyTorch was installed without CUDA support (CPU-only build)")
                else:
                    diagnostics.append(f"PyTorch CUDA build version: {pytorch_cuda}")
                
                # Check if nvidia-smi is available (driver check)
                try:
                    import subprocess
                    nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
                    if nvidia_smi.returncode == 0:
                        diagnostics.append("NVIDIA driver is installed (nvidia-smi works)")
                        # Try to extract CUDA version from nvidia-smi
                        output = nvidia_smi.stdout.decode('utf-8', errors='ignore')
                        import re
                        match = re.search(r'CUDA Version:\s*(\d+\.\d+)', output)
                        if match:
                            driver_cuda = match.group(1)
                            diagnostics.append(f"Driver CUDA version: {driver_cuda}")
                            if pytorch_cuda:
                                if driver_cuda != pytorch_cuda:
                                    diagnostics.append(f"⚠ Version mismatch: Driver CUDA {driver_cuda} vs PyTorch CUDA {pytorch_cuda}")
                    else:
                        diagnostics.append("⚠ nvidia-smi not working (driver issue?)")
                except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                    diagnostics.append("⚠ nvidia-smi not found (NVIDIA driver may not be installed)")
                
                # Check environment variables
                if 'CUDA_PATH' in os.environ:
                    diagnostics.append(f"CUDA_PATH: {os.environ['CUDA_PATH']}")
                else:
                    diagnostics.append("CUDA_PATH environment variable not set")
                
                if log_widget:
                    log(log_widget, "⚠ Using CPU (GPU not available)", root)
                    log(log_widget, "Diagnostics:", root)
                    for diag in diagnostics:
                        log(log_widget, f"  - {diag}", root)
                    log(log_widget, "To fix: Run 'python setup_gpu.py' or install PyTorch with CUDA support", root)
        except ImportError:
            if log_widget:
                log(log_widget, "⚠ PyTorch not found - using default device", root)
        except Exception as e:
            # Log the actual error for debugging
            if log_widget:
                log(log_widget, f"⚠ GPU detection error: {e}", root)
                log(log_widget, "Falling back to CPU mode", root)
        
        # Estimate model loading work (small portion)
        model_loading_work = audio_duration * 0.05
        eta_tracker.update(model_loading_work, 'model_loading')
        if progress_callback:
            progress_callback(eta_tracker.get_progress_percent(), eta_tracker, 'model_loading')

        # Stage 2: Transcription (85% of total work)
        if log_widget: log(log_widget, "→ Transcribing anime audio...", root)
        # Anime-specific Japanese context prompt optimized for:
        # - Fast speech (早口)
        # - Overlapping voices (重なる声)
        # - Slang and casual speech (スラング、カジュアルな会話)
        # - Shouting and whispering (叫び声、ささやき)
        # - Background music and SFX (BGM、効果音)
        initial_prompt = "これはアニメの音声です。早口の会話、重なる声、スラングやカジュアルな表現、叫び声やささやき、BGMや効果音が含まれています。すべての台詞を正確に聞き取ってください。"
        target_language = "ja"
        task = "transcribe"
        
        # Advanced parameters optimized for ANIME audio transcription (speed-optimized)
        # Balanced for speed while maintaining accuracy for:
        # - Fast speech: Optimized beam_size and focused temperature range
        # - Overlapping voices: More lenient thresholds
        # - Slang/casual speech: Better context awareness
        # - Shouting/whispering: Lower no_speech_threshold to catch quiet sounds
        # - Background music & SFX: Higher compression_ratio_threshold to filter repetitive patterns
        transcribe_params = {
            "language": "ja",  # Source language is always Japanese
            "task": task,
            "initial_prompt": initial_prompt,
            # Temperature sampling: Focused range for speed (reduced from 7 to 4 samples)
            # Covers key ranges: greedy (0.0), conservative (0.2), moderate (0.4), diverse (0.6)
            "temperature": (0.0, 0.2, 0.4, 0.6),
            # Optimized beam search: Reduced from 7 to 5 for ~30% speed improvement
            # Still maintains good accuracy for fast/ambiguous speech
            "beam_size": 5,  # Balanced: good accuracy with better speed
            # Condition on previous text for better context awareness (crucial for anime dialogue)
            # Kept True as it's important for context but has minimal speed impact
            "condition_on_previous_text": True,
        }
        
        # Anime-specific optional parameters
        # These are tuned specifically for anime audio characteristics
        optional_params = {
            # Higher compression ratio threshold to filter out repetitive background music/SFX
            # Anime often has looping BGM that can be mistaken for speech
            "compression_ratio_threshold": 3.0,  # Increased from 2.4 to filter music patterns
            
            # More lenient log probability threshold for overlapping voices and fast speech
            # Allows capturing dialogue even when confidence is slightly lower
            "logprob_threshold": -0.8,  # Increased from -1.0 to catch fast/overlapping speech
            
            # Lower no_speech_threshold to catch whispers and quiet dialogue
            # Anime often has quiet moments that need to be captured
            # Also helps with background music - lower threshold means less aggressive filtering
            "no_speech_threshold": 0.3,  # Lowered from 0.6 to catch whispers and handle BGM better
            
            # Word timestamps for better accuracy with fast speech
            # Helps align text with rapid dialogue
            "word_timestamps": False,  # Set to True if you need word-level timestamps (slower)
        }
        
        # Try to add optional parameters (they may not be in all Whisper versions)
        for key, value in optional_params.items():
            transcribe_params[key] = value
        
        if log_widget: 
            log(log_widget, f"→ Starting transcription (this may take a while for long files)...", root)
        
        # Start transcription monitoring thread
        transcription_start_time = time.time()
        transcription_monitor_active = threading.Event()
        transcription_monitor_active.set()
        
        # Shared state for progress estimation
        progress_state = {'estimated_speed': 0.2, 'last_elapsed': 0.0}
        
        def monitor_transcription_progress():
            """Monitor transcription progress by estimating based on elapsed time"""
            # Estimate transcription speed: typically 0.1-0.5x realtime for large models
            # Start conservative, will adjust based on actual progress
            estimated_speed = progress_state['estimated_speed']
            
            while transcription_monitor_active.is_set():
                elapsed = time.time() - transcription_start_time
                
                # Gradually increase estimated speed as time passes (adaptive learning)
                # This helps the ETA become more accurate over time
                if elapsed > 5.0:  # After 5 seconds, start learning
                    # Estimate that we're making progress, so speed should increase slightly
                    # But cap it at reasonable maximum (0.5x for large models)
                    max_speed = 0.5 if 'large' in model_size else 1.0
                    estimated_speed = min(max_speed, estimated_speed * 1.01)  # Very gradual increase
                    progress_state['estimated_speed'] = estimated_speed
                
                # Estimate completed work based on elapsed time and speed
                # Use exponential approach: more progress early, then steady
                # This mimics how Whisper processes audio (faster at start, then steady)
                progress_factor = min(0.85, 0.85 * (1 - (0.9 ** (elapsed / 10.0))))
                estimated_completed = audio_duration * progress_factor
                
                # Update tracker
                total_completed = audio_duration * 0.05 + estimated_completed
                eta_tracker.update(total_completed, 'transcription')
                
                if progress_callback:
                    progress_callback(eta_tracker.get_progress_percent(), eta_tracker, 'transcription')
                
                progress_state['last_elapsed'] = elapsed
                time.sleep(0.5)  # Update every 0.5 seconds
        
        monitor_thread = threading.Thread(target=monitor_transcription_progress, daemon=True)
        monitor_thread.start()
        
        try:
            # Try transcription with all parameters first
            try:
                result = model.transcribe(audio_path, **transcribe_params)
            except TypeError as e:
                # If some parameters aren't supported, retry with only core parameters
                if log_widget:
                    log(log_widget, "→ Some advanced parameters not supported, using core parameters...", root)
                core_params = {
                    "language": "ja",
                    "task": task,
                    "initial_prompt": initial_prompt,
                    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    "beam_size": 5,
                    "condition_on_previous_text": True,
                }
                result = model.transcribe(audio_path, **core_params)
            
            transcription_monitor_active.clear()  # Stop monitoring
            
            # Calculate actual transcription time and update speed estimate
            actual_transcription_time = time.time() - transcription_start_time
            if actual_transcription_time > 0:
                actual_speed = audio_duration / actual_transcription_time
                # Update the tracker with actual speed for final ETA accuracy
                progress_state['estimated_speed'] = actual_speed
            
            # Update with actual completion (85% of work done)
            transcription_work = audio_duration * 0.85
            eta_tracker.update(audio_duration * 0.05 + transcription_work, 'transcription')
            
            if log_widget: 
                log(log_widget, f"✓ Transcription completed, processing results...", root)
        except Exception as transcribe_error:
            transcription_monitor_active.clear()
            if log_widget: log(log_widget, f"❌ Transcription error: {transcribe_error}", root)
            raise

        if progress_callback:
            progress_callback(eta_tracker.get_progress_percent(), eta_tracker, 'transcription')

        segments = result["segments"]

        # Stage 3: File Writing (10% of total work)
        if progress_callback:
            progress_callback(eta_tracker.get_progress_percent(), eta_tracker, 'file_writing')

        # Use Japanese normalization
        normalize_func = normalize_japanese

        # --- Segmented TXT ---
        txt_path = f"{output_prefix}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                start = sec_to_srt_time(seg["start"])
                end = sec_to_srt_time(seg["end"])
                text = normalize_func(seg["text"])
                f.write(f"[{start} --> {end}] {text}\n")

        if log_widget: log(log_widget, f"✓ Segmented TXT saved: {txt_path}", root)
        
        # Update progress for TXT writing (5% of file writing stage)
        file_writing_progress = audio_duration * 0.95
        eta_tracker.update(file_writing_progress, 'file_writing')
        if progress_callback:
            progress_callback(eta_tracker.get_progress_percent(), eta_tracker, 'file_writing')

        # --- SRT ---
        srt_path = f"{output_prefix}.srt"
        create_srt_from_segments(segments, srt_path, log_widget, root)

        # Complete!
        eta_tracker.update(audio_duration, 'file_writing')
        if progress_callback:
            progress_callback(100.0, eta_tracker, 'complete')

        if log_widget:
            log(log_widget, f"=== TRANSCRIPTION COMPLETE ===", root)

    except Exception as e:
        if progress_callback:
            progress_callback(0, None, 'error')
        if log_widget: log(log_widget, f"❌ ERROR: {e}", root)
        raise

# ======================================================
# ------------------------- GUI ------------------------
# ======================================================

class App:
    def __init__(self, root):
        self.root = root
        root.title("Anime Audio Transcriber")

        # Model selection
        model_frame = Label(root, text="Whisper Model:", font=("Arial", 10))
        model_frame.pack(pady=5)
        
        self.model_var = StringVar(value="large")
        self.model_combo = ttk.Combobox(
            root, 
            textvariable=self.model_var,
            values=[
                "tiny", "tiny.en",
                "base", "base.en",
                "small", "small.en",
                "medium", "medium.en",
                "large", "large-v1", "large-v2", "large-v3"
            ],
            state="readonly",
            width=15
        )
        self.model_combo.pack(pady=5)

        self.label = Label(root, text="Select an anime audio file to transcribe", font=("Arial", 12))
        self.label.pack(pady=10)
        
        # Drag and drop area
        self.drop_label = Label(
            root, 
            text="📁 Drag & Drop audio file here\nor click 'Choose Audio File'",
            font=("Arial", 10),
            relief="ridge",
            borderwidth=2,
            padx=20,
            pady=20,
            bg="#f0f0f0"
        )
        self.drop_label.pack(pady=10, padx=20, fill="x")
        
        # Enable drag and drop
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)

        self.select_button = Button(root, text="Choose Audio File", command=self.load_file, width=20)
        self.select_button.pack(pady=5)

        self.transcribe_button = Button(root, text="Start Transcription", command=self.start_transcription, width=20)
        self.transcribe_button.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(root, mode='determinate', length=400)
        self.progress.pack(pady=5, padx=20, fill="x")
        
        # Progress label with ETA
        self.progress_label = Label(root, text="Ready", font=("Arial", 9), fg="#666666")
        self.progress_label.pack(pady=2)

        self.log = Text(root, height=15, width=70)
        self.log.pack(padx=10, pady=10)

        self.audio_path = None
        self.is_processing = False
        self.eta_tracker = None
        self.progress_update_timer = None

    def on_drop(self, event):
        """Handle drag and drop file"""
        try:
            files = self.root.tk.splitlist(event.data)
            if files:
                file_path = files[0].strip('{}').strip('"')
                # Handle Windows paths that might have braces
                if file_path.startswith('{') and file_path.endswith('}'):
                    file_path = file_path[1:-1]
                # Handle file paths with spaces or special characters
                file_path = file_path.strip()
                if os.path.exists(file_path) and self.is_valid_audio_file(file_path):
                    self.audio_path = file_path
                    log(self.log, f"Selected: {self.audio_path}")
                    self.drop_label.config(text=f"📁 {os.path.basename(self.audio_path)}", bg="#e8f5e9")
                else:
                    log(self.log, f"❌ Invalid file type or file not found. Please select an audio file.")
                    self.drop_label.config(bg="#ffebee")
        except Exception as e:
            log(self.log, f"❌ Error handling dropped file: {e}")
            self.drop_label.config(bg="#ffebee")

    def is_valid_audio_file(self, file_path):
        """Check if file is a valid audio file"""
        valid_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
        return any(file_path.lower().endswith(ext) for ext in valid_extensions)

    def load_file(self):
        file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg")])
        if file:
            self.audio_path = file
            log(self.log, f"Selected: {file}")
            self.drop_label.config(text=f"📁 {os.path.basename(self.audio_path)}", bg="#e8f5e9")

    def start_transcription(self):
        if not self.audio_path:
            log(self.log, "Please select an audio file first.")
            return
        
        if not os.path.exists(self.audio_path):
            log(self.log, f"❌ Error: File not found: {self.audio_path}")
            return
        
        if self.is_processing:
            log(self.log, "Transcription already in progress. Please wait...")
            return

        self.is_processing = True
        self.progress['value'] = 0
        self.progress_label.config(text="Initializing...", fg="#666666")
        self.transcribe_button.config(state="disabled")
        self.select_button.config(state="disabled")
        self.model_combo.config(state="disabled")
        
        # Reset progress bar to determinate mode
        if self.progress['mode'] == 'indeterminate':
            self.progress.stop()
            self.progress.config(mode='determinate')
        
        # Determine output prefix
        output_prefix = os.path.splitext(self.audio_path)[0] + "_jp"
        model_size = self.model_var.get()

        # Start progress update loop
        self.start_progress_updates()

        # Run in background thread so GUI doesn't freeze
        thread = threading.Thread(
            target=self.transcribe_with_callback,
            args=(self.audio_path, output_prefix, model_size),
            daemon=True
        )
        thread.start()

    def update_progress(self, progress_percent, eta_tracker=None, stage=None):
        """
        Thread-safe progress bar update with ETA tracking
        
        Args:
            progress_percent: Progress percentage (0-100)
            eta_tracker: ETATracker instance (optional)
            stage: Current stage name (optional)
        """
        if eta_tracker:
            self.eta_tracker = eta_tracker
        
        # Update progress bar
        if self.eta_tracker and self.eta_tracker.should_use_indeterminate():
            # Switch to indeterminate mode during high uncertainty
            if self.progress['mode'] != 'indeterminate':
                self.progress.config(mode='indeterminate')
                self.progress.start(10)  # Start animation
        else:
            # Use determinate mode with actual progress
            if self.progress['mode'] != 'determinate':
                self.progress.stop()
                self.progress.config(mode='determinate')
            
            self.progress['value'] = progress_percent
        
        # Update progress label with ETA information
        self.update_progress_label()
        
        self.root.update_idletasks()
    
    def update_progress_label(self):
        """Update the progress label with ETA and status information"""
        if not self.eta_tracker:
            self.progress_label.config(text="Ready")
            return
        
        if self.eta_tracker.is_complete():
            elapsed = self.eta_tracker.get_elapsed_time()
            if elapsed < 60:
                elapsed_str = f"{int(elapsed)}s"
            elif elapsed < 3600:
                elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
            else:
                elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m"
            self.progress_label.config(text=f"Complete! (took {elapsed_str})", fg="#4caf50")
            return
        
        # Build status string
        progress_pct = self.eta_tracker.get_progress_percent()
        eta_str = self.eta_tracker.get_eta_string()
        speed_str = self.eta_tracker.get_speed_string()
        stage_name = self.eta_tracker.current_stage or "Processing"
        
        # Format stage name nicely
        stage_display = {
            'model_loading': 'Loading Model',
            'transcription': 'Processing',
            'file_writing': 'Saving Files',
            'complete': 'Complete',
            'error': 'Error'
        }.get(stage_name, stage_name.replace('_', ' ').title())
        
        if self.eta_tracker.should_use_indeterminate():
            status_text = f"{stage_display}... ({progress_pct:.1f}%)"
            self.progress_label.config(text=status_text, fg="#666666")
        else:
            status_text = f"{stage_display}: {progress_pct:.1f}% | ETA: {eta_str} | Speed: {speed_str}"
            self.progress_label.config(text=status_text, fg="#2196f3")
    
    def start_progress_updates(self):
        """Start periodic progress label updates"""
        if self.progress_update_timer:
            self.root.after_cancel(self.progress_update_timer)
        
        def update_loop():
            if self.is_processing and self.eta_tracker:
                self.update_progress_label()
                self.progress_update_timer = self.root.after(500, update_loop)  # Update every 500ms
        
        self.progress_update_timer = self.root.after(500, update_loop)
    
    def stop_progress_updates(self):
        """Stop periodic progress label updates"""
        if self.progress_update_timer:
            self.root.after_cancel(self.progress_update_timer)
            self.progress_update_timer = None

    def transcribe_with_callback(self, audio_path, output_prefix, model_size):
        """Wrapper to handle transcription with GUI updates"""
        try:
            transcribe_audio(
                audio_path, 
                output_prefix, 
                model_size,
                self.log, 
                lambda p, e, s: self.root.after(0, self.update_progress, p, e, s),
                None,  # ETA tracker will be created inside transcribe_audio
                self.root
            )
        except Exception as e:
            # Log any unexpected errors
            log(self.log, f"❌ Unexpected error: {e}", self.root)
            self.root.after(0, lambda: self.update_progress(0, None, 'error'))
        finally:
            # Re-enable buttons after transcription completes (success or failure)
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        """Reset UI after transcription/translation"""
        self.is_processing = False
        self.stop_progress_updates()
        self.eta_tracker = None
        
        # Ensure progress bar is in determinate mode
        if self.progress['mode'] == 'indeterminate':
            self.progress.stop()
            self.progress.config(mode='determinate')
        
        self.transcribe_button.config(state="normal")
        self.select_button.config(state="normal")
        self.model_combo.config(state="readonly")


# ======================================================
# ----------------------- MAIN -------------------------
# ======================================================

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = App(root)
    root.mainloop()
