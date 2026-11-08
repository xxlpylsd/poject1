"""Runtime environment normalization done before NumPy/PyTorch imports."""

from __future__ import annotations

import os


def normalize_thread_environment() -> None:
    """Replace malformed OpenMP thread-count variables by a safe value."""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            if int(value) <= 0:
                raise ValueError
        except ValueError:
            os.environ[name] = "1"
