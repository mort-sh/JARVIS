"""
Direct CUDA Test Script

This script attempts to download and test PyTorch with CUDA support
from a direct wheel URL without using pip.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import site
import importlib.util

# Create a temporary directory for downloading files
temp_dir = tempfile.mkdtemp()
print(f"Created temporary directory: {temp_dir}")

try:
    # Function to install a package from a wheel URL
    def install_from_wheel(url, package_name):
        try:
            # Download the wheel
            wheel_path = os.path.join(temp_dir, f"{package_name}.whl")
            print(f"Downloading {url}")

            # Use curl to download the wheel
            subprocess.check_call(
                ["curl", "-L", url, "--output", wheel_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Install the wheel
            print(f"Installing {package_name}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--force-reinstall", wheel_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            print(f"Successfully installed {package_name}")
            return True
        except Exception as e:
            print(f"Error installing {package_name}: {e}")
            return False

    # Print system information
    print("\nSystem Information:")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")

    # First, check if PyTorch is already installed with CUDA support
    try:
        import torch
        print("\nPyTorch is already installed:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available in the current PyTorch installation.")

    except ImportError:
        print("\nPyTorch is not installed. Attempting direct installation...")

        # URLs for PyTorch with CUDA 12.8 support for Python 3.12
        # Note: These URLs might need to be updated based on availability
        torch_url = "https://download.pytorch.org/whl/cu128/torch-2.3.0%2Bcu128-cp312-cp312-win_amd64.whl"

        if install_from_wheel(torch_url, "torch"):
            # Try importing the newly installed PyTorch
            import torch
            print("\nPyTorch direct installation results:")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU devices: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
            else:
                print("CUDA is not available despite installation with CUDA support.")
                print("\nPossible reasons:")
                print("1. NVIDIA drivers are not properly installed or outdated")
                print("2. CUDA Toolkit is not installed or not compatible")
                print("3. PyTorch version is not compatible with your installed CUDA version")
        else:
            print("Failed to install PyTorch with CUDA support directly.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up the temporary directory
    print(f"\nCleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)
