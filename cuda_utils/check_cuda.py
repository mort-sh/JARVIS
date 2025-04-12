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
else:
    print("CUDA version: N/A")
    print("GPU devices: N/A")
    print("GPU name: N/A")
