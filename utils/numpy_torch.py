from __future__ import annotations
import numpy as np
import torch

def as_tensor(value):
    if not isinstance(value, np.ndarray):
        return value
    array = np.ascontiguousarray(value)
    try:
        return torch.from_numpy(array)
    except TypeError:
        return torch.from_dlpack(array)
