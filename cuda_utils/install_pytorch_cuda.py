"""
PyTorch CUDA Installation Script

This script installs PyTorch with CUDA support in a dedicated test environment
and verifies that it's working properly.
"""

import os
import sys
import subprocess
import shutil
import tempfile

# Configuration
CUDA_VERSION = "cu128"  # CUDA 12.8
TEST_ENV_NAME = "pytorch_cuda_test_env"

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

def create_test_env():
    """Create a temporary environment for testing PyTorch with CUDA"""
    # Create a temporary directory for the test environment
    temp_dir = tempfile.mkdtemp(prefix="pytorch_cuda_test_")
    print(f"\nCreated temporary directory: {temp_dir}")

    # Create a virtual environment in the temporary directory
    if not run_command(f"python -m venv {os.path.join(temp_dir, TEST_ENV_NAME)}",
                      "Creating test virtual environment"):
        shutil.rmtree(temp_dir)
        return None

    # Determine the path to python in the test environment
    if os.name == 'nt':  # Windows
        python_path = os.path.join(temp_dir, TEST_ENV_NAME, "Scripts", "python.exe")
    else:  # Unix-like
        python_path = os.path.join(temp_dir, TEST_ENV_NAME, "bin", "python")

    # Upgrade pip in the test environment
    if not run_command(f'"{python_path}" -m pip install --upgrade pip',
                      "Upgrading pip in test environment"):
        shutil.rmtree(temp_dir)
        return None

    return temp_dir, python_path

def install_pytorch(python_path):
    """Install PyTorch with CUDA support in the test environment"""
    pytorch_cmd = (f'"{python_path}" -m pip install torch torchvision torchaudio '
                  f'--index-url https://download.pytorch.org/whl/{CUDA_VERSION}')

    return run_command(pytorch_cmd, "Installing PyTorch with CUDA support")

def verify_cuda_support(python_path):
    """Verify that CUDA support is available in PyTorch"""
    # Create a temporary verification script
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

    fd, verify_path = tempfile.mkstemp(suffix=".py")
    os.close(fd)

    with open(verify_path, "w") as f:
        f.write(verify_script)

    print("\nVerifying CUDA support...")
    result = run_command(f'"{python_path}" "{verify_path}"', check=False)

    # Clean up
    os.remove(verify_path)

    return result

def update_pyproject_toml():
    """Update the pyproject.toml file with PyTorch CUDA configuration"""
    pyproject_path = os.path.join(os.getcwd(), "pyproject.toml")

    if not os.path.exists(pyproject_path):
        print("\nWarning: pyproject.toml not found")
        return

    print("\nUpdating pyproject.toml with CUDA configuration...")

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
    print("PyTorch CUDA Test Installation")
    print("=" * 70)

    # Create temporary test environment
    result = create_test_env()
    if not result:
        print("Failed to create test environment")
        return

    temp_dir, python_path = result

    try:
        # Install PyTorch with CUDA support
        if not install_pytorch(python_path):
            print("Failed to install PyTorch with CUDA support")
            return

        # Verify CUDA support
        verify_cuda_support(python_path)

        # Update pyproject.toml
        update_pyproject_toml()

        print("\n" + "=" * 70)
        print("PyTorch CUDA Test Completed")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Use the following pip command to install PyTorch with CUDA support in your project:")
        print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{CUDA_VERSION}")
        print("\n2. Or use UV if you have it installed:")
        print(f"   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{CUDA_VERSION}")

    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
