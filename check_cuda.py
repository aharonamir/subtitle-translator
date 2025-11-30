import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import sys

print(f"Python version: {sys.version}")
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
else:
    print("\nPossible reasons for CUDA not being available:")
    print("1. NVIDIA GPU drivers not installed")
    print("2. CUDA toolkit not installed")
    print("3. PyTorch installed without CUDA support")
    print("\nTo install PyTorch with CUDA support, run:")
    print("conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
