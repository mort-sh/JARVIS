"""
CUDA Setup Script for PyTorch

This script documents the process of setting up PyTorch with CUDA support.
It can be used to reinstall PyTorch with CUDA support if needed.

Requirements:
- CUDA Toolkit installed (v12.6 recommended)
- Compatible NVIDIA GPU drivers
- UV package manager

Process:
1. Uninstall existing PyTorch packages
2. Install PyTorch with CUDA support using the correct index URL
3. Verify CUDA is available
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and print the output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def setup_pytorch_cuda():
    """Setup PyTorch with CUDA support"""
    # Get the virtual environment path
    venv_path = os.path.dirname(os.path.dirname(sys.executable))
    print(f"Virtual environment: {venv_path}")

    # Uninstall existing PyTorch packages
    print("\n1. Uninstalling existing PyTorch packages...")
    run_command(f"{sys.executable} -m uv pip uninstall -y torch torchvision torchaudio")

    # Install PyTorch with CUDA support
    print("\n2. Installing PyTorch with CUDA support...")
    cuda_version = "cu126"  # CUDA 12.6
    index_url = f"https://download.pytorch.org/whl/{cuda_version}"

    # Install PyTorch packages with CUDA support
    run_command(f"{sys.executable} -m uv pip install torch torchvision torchaudio --index-url {index_url}")

    # Verify CUDA is available
    print("\n3. Verifying CUDA is available...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Check your installation.")
    except ImportError:
        print("Failed to import torch. Check your installation.")

if __name__ == "__main__":
    setup_pytorch_cuda()
