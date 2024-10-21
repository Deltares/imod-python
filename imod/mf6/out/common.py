import pathlib
from typing import Any, BinaryIO, Dict, List, Union

import numpy as np
import struct

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


def get_first_header_advanced_package(
    headers: Dict[str, List[Any]],
) -> Any:
    for key, header_list in headers.items():
        # multimodels have a gwf-gwf budget for flow-ja-face between domains
        if "flow-ja-face" not in key and "gwf_" in key:
            return header_list[0]
    return None


def read_name_dvs(path: FilePath) -> str:
    """
    Reads variable name from first header in dependent variable file.
    """
    with open(path, "rb") as f:
        f.seek(24)
        name = struct.unpack("16s", f.read(16))[0]
    return name.decode().strip()


def read_times_dvs(path: FilePath, ntime: int, indices: np.ndarray) -> FloatArray:
    """
    Reads all total simulation times.
    """
    times = np.empty(ntime, dtype=np.float64)

    # Compute how much to skip to the next timestamp
    start_of_header = 16
    rest_of_header = 28
    data_single_layer = indices.size * 8
    nskip = rest_of_header + data_single_layer + start_of_header

    with open(path, "rb") as f:
        f.seek(start_of_header)
        for i in range(ntime):
            times[i] = struct.unpack("d", f.read(8))[0]  # total simulation time
            f.seek(nskip, 1)
    return times
