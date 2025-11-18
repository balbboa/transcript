#!/usr/bin/env python3
"""
Automatic GPU/CUDA detection and PyTorch installation script
Detects your GPU and CUDA version, then installs the correct PyTorch version
"""

import subprocess
import sys
import re

def run_command(cmd, shell=True):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=10)
        return result.stdout.strip(), result.returncode == 0
    except Exception as e:
        return "", False

def detect_nvidia_gpu():
    """Detect if NVIDIA GPU is present"""
    print("🔍 Checking for NVIDIA GPU...")
    output, success = run_command("nvidia-smi")
    if success and "NVIDIA" in output:
        print("✓ NVIDIA GPU detected!")
        return True
    else:
        print("⚠ No NVIDIA GPU detected (or nvidia-smi not available)")
        return False

def detect_cuda_version():
    """Detect CUDA version from nvidia-smi"""
    print("🔍 Detecting CUDA version...")
    output, success = run_command("nvidia-smi")
    if success:
        # Look for "CUDA Version: X.Y" in nvidia-smi output
        match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', output)
        if match:
            major, minor = match.groups()
            cuda_version = f"{major}.{minor}"
            print(f"✓ Detected CUDA version: {cuda_version}")
            return cuda_version
    
    # Try alternative method: check CUDA_PATH
    output, success = run_command("echo %CUDA_PATH%")
    if success and output and output != "%CUDA_PATH%":
        print(f"✓ Found CUDA_PATH: {output}")
        # Try to extract version from path
        match = re.search(r'(\d+)\.(\d+)', output)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
    
    print("⚠ Could not detect CUDA version automatically")
    return None

def get_pytorch_install_command(cuda_version=None):
    """Get the appropriate PyTorch installation command"""
    if cuda_version:
        major, minor = map(int, cuda_version.split('.'))
        
        # CUDA 13.0+
        if major >= 13:
            print("→ Using CUDA 13.0 build (or latest compatible)")
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130"
        
        # CUDA 12.4
        elif major == 12 and minor >= 4:
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
        
        # CUDA 12.1
        elif major == 12 and minor >= 1:
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        
        # CUDA 11.8
        elif major == 11 and minor >= 8:
            return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    # Fallback: try CUDA 12.4 (most compatible)
    print("→ Using CUDA 12.4 build (backward compatible)")
    return "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"

def install_pytorch(cuda_version=None):
    """Install PyTorch with appropriate CUDA support"""
    print("\n📦 Installing PyTorch...")
    cmd = get_pytorch_install_command(cuda_version)
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print("✓ PyTorch installed successfully!")
        return True
    else:
        print("❌ PyTorch installation failed!")
        return False

def verify_installation():
    """Verify PyTorch and CUDA installation"""
    print("\n🔍 Verifying installation...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠ CUDA not available (CPU mode)")
            print("  This might mean:")
            print("  - PyTorch was installed without CUDA support")
            print("  - CUDA drivers are not properly installed")
            print("  - GPU is not compatible")
            return False
    except ImportError:
        print("❌ PyTorch not found!")
        return False
    except Exception as e:
        print(f"❌ Error verifying installation: {e}")
        return False

def install_requirements():
    """Install other requirements from requirements.txt"""
    print("\n📦 Installing other requirements...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if result.returncode == 0:
        print("✓ Requirements installed successfully!")
        return True
    else:
        print("⚠ Some requirements may have failed to install")
        return False

def main():
    print("=" * 60)
    print("  Automatic GPU/CUDA Setup for Anime Transcriber")
    print("=" * 60)
    print()
    
    # Step 1: Detect GPU
    has_gpu = detect_nvidia_gpu()
    print()
    
    # Step 2: Detect CUDA version
    cuda_version = None
    if has_gpu:
        cuda_version = detect_cuda_version()
    print()
    
    # Step 3: Install PyTorch
    if has_gpu:
        print("🚀 GPU detected - installing PyTorch with CUDA support...")
        if not install_pytorch(cuda_version):
            print("\n⚠ GPU installation failed, trying CPU version...")
            install_pytorch(None)  # Try CPU version
    else:
        print("💻 No GPU detected - installing PyTorch CPU version...")
        install_pytorch(None)
    print()
    
    # Step 4: Install other requirements
    install_requirements()
    print()
    
    # Step 5: Verify
    if verify_installation():
        print("\n" + "=" * 60)
        print("  ✓ Setup completed successfully!")
        print("=" * 60)
        if has_gpu:
            print("\n🎉 Your GPU is ready! Transcription will be much faster.")
        else:
            print("\n💡 Tip: Install NVIDIA drivers and CUDA to enable GPU acceleration")
    else:
        print("\n" + "=" * 60)
        print("  ⚠ Setup completed with warnings")
        print("=" * 60)
        print("\nYou can still use the transcriber, but it will run on CPU (slower)")

if __name__ == "__main__":
    main()

