import pathlib
from typing import Any, BinaryIO, Dict, List, Union

import numpy as np

# Type annotations
IntArray = np.ndarray
FloatArray = np.ndarray
FilePath = Union[str, pathlib.Path]


def _grb_text(f: BinaryIO, lentxt: int = 50) -> str:
    return f.read(lentxt).decode("utf-8").strip().lower()


def _to_nan(a: FloatArray, dry_nan: bool) -> FloatArray:
    # TODO: this could really use a docstring?
    a[a == 1e30] = np.nan
    if dry_nan:
        a[a == -1e30] = np.nan
    return a


def get_header_advanced_package(
    headers: Dict[str, List[Any]],
) -> Any:
    for key, header in headers.items():
        # multimodels have a gwf-gwf budget for flow-ja-face between domains
        if not "flow-ja-face" in key and "gwf_" in key:
            return header[0]
    return None
