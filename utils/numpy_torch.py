"""NumPy-to-PyTorch conversion with a DLPack compatibility fallback."""

from __future__ import annotations

import numpy as np
import torch


def as_tensor(value):
    """Return ``value`` unchanged unless it is a NumPy array.

    Some PyTorch wheels expose a broken ``torch.from_numpy`` bridge even when
    the NumPy version itself is compatible.  DLPack uses a different C API
    path, so it provides a zero-copy fallback without changing array values.
    """
    if not isinstance(value, np.ndarray):
        return value

    array = np.ascontiguousarray(value)
    try:
        return torch.from_numpy(array)
    except TypeError:
        return torch.from_dlpack(array)
