# CUDA Utilities

This directory contains utilities for setting up, verifying, and working with CUDA and PyTorch CUDA integration.

## Scripts

- **capture_cuda_info.py**: Captures detailed CUDA environment information and saves to a file
- **check_cuda.py**: Simple script to check if CUDA is available and accessible to Python
- **cuda_setup.py**: Installs PyTorch with CUDA support
- **direct_install_pytorch_cuda.py**: Direct installation of PyTorch with specific CUDA version
- **install_pytorch_cuda.py**: Interactive installer for PyTorch with CUDA
- **setup_pytorch_cuda_env.py**: Sets up the environment for PyTorch CUDA
- **simple_cuda_check.py**: Minimal script to verify CUDA is working
- **test_cuda_direct.py**: Test script for direct CUDA operations
- **verify_cuda.py**: Comprehensive CUDA verification script (referenced in docs)

## Usage

For most users, running the `verify_cuda.py` script is the best way to test CUDA availability:

```bash
python cuda_utils/verify_cuda.py
```

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

## Installation Scripts

For installing PyTorch with CUDA support:

1. **Interactive Installation**:
   ```bash
   python cuda_utils/install_pytorch_cuda.py
   ```

2. **Direct Installation** (with CUDA 12.6):
   ```bash
   python cuda_utils/direct_install_pytorch_cuda.py
   ```

See the [CUDA Setup Guide](../docs/CUDA_SETUP.md) for more detailed information.
