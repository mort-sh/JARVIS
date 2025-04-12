"""
PyTorch CUDA Environment Setup Script

This script automates the complete process of setting up a new virtual environment
with PyTorch and CUDA support.

Steps:
1. Create a new virtual environment
2. Install PyTorch with CUDA support
3. Verify CUDA is available and working
"""

import os
import sys
import subprocess
import shutil

# Configuration
VENV_PATH = os.path.join(os.getcwd(), ".venv")
CUDA_VERSION = "cu128"  # CUDA 12.8

def run_command(cmd, description=None, check=True):
    """Run a command and print its output"""
    if description:
        print(f"\n{description}...")

    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, text=True,
                           capture_output=True)

    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")

    return result.returncode == 0

def setup_environment():
    """Create and set up the virtual environment"""
    # Check if Python is available
    py_version_result = subprocess.run(["python", "--version"],
                                        capture_output=True, text=True)
    if py_version_result.returncode != 0:
        print("Error: Python is not available in PATH")
        return False

    print(f"Using Python: {py_version_result.stdout.strip()}")

    # Remove existing virtual environment if it exists
    if os.path.exists(VENV_PATH):
        print(f"\nRemoving existing virtual environment at {VENV_PATH}")
        try:
            shutil.rmtree(VENV_PATH)
        except Exception as e:
            print(f"Error removing virtual environment: {e}")
            return False

    # Create new virtual environment
    if not run_command("python -m venv .venv", "Creating new virtual environment"):
        return False

    # Determine the path to pip in the virtual environment
    if os.name == 'nt':  # Windows
        python_path = os.path.join(VENV_PATH, "Scripts", "python.exe")
        pip_path = os.path.join(VENV_PATH, "Scripts", "pip.exe")
    else:  # Unix-like
        python_path = os.path.join(VENV_PATH, "bin", "python")
        pip_path = os.path.join(VENV_PATH, "bin", "pip")

    # Upgrade pip in the virtual environment
    if not run_command(f'"{python_path}" -m pip install --upgrade pip',
                      "Upgrading pip in virtual environment"):
        return False

    # Install PyTorch with CUDA support
    pytorch_cmd = (f'"{python_path}" -m pip install torch torchvision torchaudio '
                  f'--index-url https://download.pytorch.org/whl/{CUDA_VERSION}')

    if not run_command(pytorch_cmd, "Installing PyTorch with CUDA support"):
        return False

    return True

def verify_cuda_support():
    """Verify that CUDA support is available in PyTorch"""
    # Determine the path to python in the virtual environment
    if os.name == 'nt':  # Windows
        python_path = os.path.join(VENV_PATH, "Scripts", "python.exe")
    else:  # Unix-like
        python_path = os.path.join(VENV_PATH, "bin", "python")

    # Create and run a verification script
    verify_script = """
import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name:", torch.cuda.get_device_name(i))

    # Simple CUDA operation test
    print("\\nRunning simple CUDA test...")
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = torch.tensor([4.0, 5.0, 6.0]).cuda()
    z = x + y
    print(f"CUDA Tensor Addition Test: {x} + {y} = {z}")
    print("CUDA test completed successfully!")
else:
    print("CUDA version: N/A")
    print("GPU devices: N/A")
    print("GPU name: N/A")

    print("\\nCUDA is not available. Possible reasons:")
    print("1. NVIDIA drivers are not properly installed or outdated")
    print("2. CUDA Toolkit is not installed or not compatible")
    print("3. PyTorch was not installed with CUDA support")
"""

    verify_path = os.path.join(os.getcwd(), "verify_cuda.py")
    with open(verify_path, "w") as f:
        f.write(verify_script)

    print("\nVerifying CUDA support...")
    run_command(f'"{python_path}" verify_cuda.py', check=False)

    # Clean up
    if os.path.exists(verify_path):
        os.remove(verify_path)

def update_pyproject_toml():
    """Update the pyproject.toml file with PyTorch CUDA configuration"""
    pyproject_path = os.path.join(os.getcwd(), "pyproject.toml")

    if not os.path.exists(pyproject_path):
        print("\nWarning: pyproject.toml not found")
        return

    print("\nUpdating pyproject.toml...")

    with open(pyproject_path, "r") as f:
        content = f.read()

    # Update or add PyTorch CUDA configuration
    updated = False

    # Check if [tool.uv.sources.pytorch] section exists
    if "[tool.uv.sources.pytorch]" in content:
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "[tool.uv.sources.pytorch]":
                # Find the url line and update it
                for j in range(i, min(i+5, len(lines))):
                    if lines[j].strip().startswith("url ="):
                        lines[j] = f'url = "https://download.pytorch.org/whl/{CUDA_VERSION}"'
                        updated = True
                        break

        # Check if [[tool.uv.index]] section exists
        for i, line in enumerate(lines):
            if line.strip() == "[[tool.uv.index]]":
                # Find the url line and update it
                for j in range(i, min(i+5, len(lines))):
                    if lines[j].strip().startswith("url ="):
                        lines[j] = f'url = "https://download.pytorch.org/whl/{CUDA_VERSION}"'
                        updated = True
                        break

        if updated:
            with open(pyproject_path, "w") as f:
                f.write("\n".join(lines))
            print("Updated PyTorch CUDA configuration in pyproject.toml")
        else:
            print("Failed to update PyTorch CUDA configuration in pyproject.toml")
    else:
        # Add the sections at the end of the file
        with open(pyproject_path, "a") as f:
            f.write("\n\n[tool.uv.sources.pytorch]\n")
            f.write(f'url = "https://download.pytorch.org/whl/{CUDA_VERSION}"\n\n')
            f.write("[[tool.uv.index]]\n")
            f.write(f'url = "https://download.pytorch.org/whl/{CUDA_VERSION}"\n')
        print("Added PyTorch CUDA configuration to pyproject.toml")

def main():
    print("=" * 70)
    print("PyTorch CUDA Environment Setup")
    print("=" * 70)

    if setup_environment():
        print("\nEnvironment setup completed successfully!")
        verify_cuda_support()
        update_pyproject_toml()

        print("\n" + "=" * 70)
        print("Setup Complete!")
        print("=" * 70)
        print("\nTo activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print("    .venv\\Scripts\\activate")
        else:  # Unix-like
            print("    source .venv/bin/activate")
    else:
        print("\nEnvironment setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
