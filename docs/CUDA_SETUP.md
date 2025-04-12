# PyTorch CUDA Setup Guide

This guide explains how to set up PyTorch with CUDA support for the JARVIS project.

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (version 537.13 or higher recommended)
- CUDA Toolkit 12.6 installed
- Python 3.12 or higher

## Checking CUDA Availability

You can verify if your system has CUDA available by running:

```bash
nvidia-smi
```

This should display information about your NVIDIA GPU, driver version, and CUDA version.

## Installing PyTorch with CUDA Support

The project uses PyTorch with CUDA 12.6 support. The configuration is already set in the `pyproject.toml` file:

```toml
[tool.uv.sources.pytorch]
url = "https://download.pytorch.org/whl/cu126"

[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu126"
```

### Installation Steps

1. **Using uv (Recommended)**

   If you have [uv](https://github.com/astral-sh/uv) installed:

   ```bash
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

2. **Using pip**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

## Verifying CUDA Support

To verify that PyTorch can access CUDA, run the included verification script:

```bash
python verify_cuda.py
```

This script will display information about PyTorch, CUDA availability, and run a simple CUDA tensor operation to confirm everything is working correctly.

### Expected Output

If CUDA is properly set up, you should see output similar to:

```
Python executable: /path/to/python
Python version: 3.12.0
PyTorch version: 2.6.0+cu126
CUDA available: True
CUDA version: 12.6
GPU devices: 1
GPU 0 name: NVIDIA GeForce RTX 3080

Running simple CUDA test...
CUDA Tensor Addition Test: tensor([1., 2., 3.], device='cuda:0') + tensor([4., 5., 6.], device='cuda:0') = tensor([5., 7., 9.], device='cuda:0')
CUDA test completed successfully!
```

## Troubleshooting

If CUDA is not available, check the following:

1. Make sure NVIDIA drivers are properly installed
2. Verify CUDA Toolkit is installed and compatible
3. Ensure PyTorch was installed with CUDA support (should show "+cu126" in the version string)
4. Check if your GPU is recognized by the system

For Python 3.13 or newer versions, you may need to check for PyTorch wheel compatibility, as support might vary.

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
