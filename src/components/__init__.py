import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")