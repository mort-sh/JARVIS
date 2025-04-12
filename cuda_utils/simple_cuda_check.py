"""
Simple CUDA availability check script that writes results to a file
"""

import torch
import sys
import os

# Open a file to write the results to
with open("cuda_results.txt", "w") as f:
    f.write(f"Python executable: {sys.executable}\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"PyTorch version: {torch.__version__}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")

    if torch.cuda.is_available():
        f.write(f"CUDA version: {torch.version.cuda}\n")
        f.write(f"GPU devices: {torch.cuda.device_count()}\n")
        for i in range(torch.cuda.device_count()):
            f.write(f"GPU {i} name: {torch.cuda.get_device_name(i)}\n")
    else:
        f.write("CUDA version: N/A\n")
        f.write("GPU devices: N/A\n")
        f.write("GPU name: N/A\n")

print("CUDA information written to cuda_results.txt")
