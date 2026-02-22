import torch
import numpy as np


def as_tensor(X, device=None, dtype=torch.float32):
    """
    Convert input to torch tensor with correct device and dtype.
    """

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if not isinstance(X, torch.Tensor):
        raise TypeError("Input must be numpy array or torch tensor")

    X = X.to(dtype=dtype)

    if device is not None:
        X = X.to(device)

    if X.ndim != 2:
        raise ValueError("Input must be 2D")

    return X.contiguous()
