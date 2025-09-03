from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np

def _to_numpy(arr):
    # works for numpy or cupy arrays
    try:
        import cupy as cp  # type: ignore
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    return np.asarray(arr)

def save_arrays_npz(path: Path, arrays: Dict[str, Any], compress: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays_np = {k: _to_numpy(v) for k, v in arrays.items()}
    if compress:
        np.savez_compressed(path, **arrays_np)
    else:
        np.savez(path, **arrays_np)
