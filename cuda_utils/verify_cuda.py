"""
CUDA Verification Script

This script verifies that PyTorch can access CUDA capabilities
and runs a basic test to confirm it's working.
"""

import torch
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name:", torch.cuda.get_device_name(i))

    # Simple CUDA operation test
    print("\nRunning simple CUDA test...")
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = torch.tensor([4.0, 5.0, 6.0]).cuda()
    z = x + y
    print(f"CUDA Tensor Addition Test: {x} + {y} = {z}")
    print("CUDA test completed successfully!")
else:
    print("CUDA version: N/A")
    print("GPU devices: N/A")
    print("GPU name: N/A")

    print("\nCUDA is not available. Possible reasons:")
    print("1. NVIDIA drivers are not properly installed or outdated")
    print("2. CUDA Toolkit is not installed or not compatible")
    print("3. PyTorch was not installed with CUDA support")
