"""
Direct PyTorch CUDA Installation Script

This script installs PyTorch with CUDA support directly using the system Python
and verifies it works correctly.
"""

import os
import sys
import subprocess
import tempfile

# Configuration
CUDA_VERSION = "cu128"  # CUDA 12.8

def run_command(cmd, description=None):
    """Run a command and print its output"""
    if description:
        print(f"\n{description}...")

    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(f"Error: {result.stderr}")

    return result.returncode == 0

def install_pytorch():
    """Install PyTorch with CUDA support"""
    # First, uninstall any existing PyTorch installations
    print("\nChecking for existing PyTorch installations...")
    try:
        import torch
        print(f"Found PyTorch {torch.__version__}")
        if "cu" in torch.__version__ or torch.cuda.is_available():
            print("CUDA support already available in PyTorch!")
            return True
        else:
            print("PyTorch installed but CUDA not available. Reinstalling with CUDA support...")
    except ImportError:
        print("PyTorch not found. Installing...")

    # Uninstall existing PyTorch packages
    run_command(f"{sys.executable} -m pip uninstall -y torch torchvision torchaudio")

    # Install PyTorch with CUDA support
    pytorch_cmd = (f"{sys.executable} -m pip install torch torchvision torchaudio "
                  f"--index-url https://download.pytorch.org/whl/{CUDA_VERSION}")

    return run_command(pytorch_cmd, "Installing PyTorch with CUDA support")

def verify_cuda_support():
    """Verify that CUDA support is available in PyTorch"""
    # Create a verification script
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

    # Write the verification script to a temporary file
    fd, verify_path = tempfile.mkstemp(suffix=".py")
    os.close(fd)

    with open(verify_path, "w") as f:
        f.write(verify_script)

    print("\nVerifying CUDA support...")
    result = run_command(f'"{sys.executable}" "{verify_path}"')

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
    print("Direct PyTorch CUDA Installation")
    print("=" * 70)

    # Install PyTorch with CUDA support
    if install_pytorch():
        # Verify CUDA support
        verify_cuda_support()

        # Update pyproject.toml
        update_pyproject_toml()

        print("\n" + "=" * 70)
        print("PyTorch CUDA Installation Completed")
        print("=" * 70)
        print("\nTo reinstall in the future, use:")
        print(f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{CUDA_VERSION}")
    else:
        print("\nPyTorch installation failed. Please check the errors above.")

if __name__ == "__main__":
    main()
