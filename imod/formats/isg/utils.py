import shlex
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar, get_args

import numpy as np
import pandas as pd


def record_length(path: str | PathLike[str]) -> int:
    """
    Record length in bytes, stored in the first record as reclen * 256 + 247,
    e.g. 2295 for an 8 byte record.
    """
    return int(np.fromfile(path, dtype="<i4", count=1)[0]) // 256


def read_records(path: str | PathLike[str], dtype: np.dtype) -> np.ndarray:
    """Read a binary file of fixed length records, skipping the first record."""
    return np.fromfile(path, dtype=dtype.newbyteorder("<"), offset=dtype.itemsize)


def split_values(line: str) -> list[str]:
    """
    Values of a line of Fortran list-directed input: separated by commas
    and/or blanks, strings optionally quoted.
    """
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace += ","
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.escape = ""
    return list(lexer)


def read_rows(path: str | PathLike[str]) -> list[list[str]]:
    """Read the values of the non-blank lines of a list-directed text file."""
    with open(path, encoding="latin-1") as f:
        return [values for line in f if (values := split_values(line))]


def decode(column: np.ndarray) -> np.ndarray:
    """Decode fixed width byte strings, stripping the padding."""
    if column.dtype.kind == "S":
        return np.char.rstrip(np.char.decode(column, "latin-1"), " \x00")
    return column


def sibling_path(path: Path, ext: str) -> Path:
    """File with the same stem and extension ``ext``, in any case."""
    if path.suffix.lower() == ext.lower() and path.exists():
        return path
    for candidate in (path.with_suffix(ext.upper()), path.with_suffix(ext.lower())):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{path.with_suffix(ext)} not found")


def record_index(first: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Zero-based indices of ``count`` records from the one-based ``first``."""
    count = np.asarray(count, dtype=np.int64)
    start = np.repeat(np.asarray(first, dtype=np.int64) - 1, count)
    offset = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    return start + offset


def to_datetime(idate: np.ndarray, ctime: np.ndarray | None = None) -> pd.DatetimeIndex:
    """
    Datetimes from IDATE (yyyymmdd) and CTIME (hh:mm:ss); a missing or blank
    CTIME is midnight, an invalid date is NaT.
    """
    time = np.char.strip(ctime) if ctime is not None else np.full(idate.shape, "")
    time = np.where(time == "", "00:00:00", time)
    text = np.char.add(idate.astype(str), time)
    return pd.to_datetime(text, format="%Y%m%d%H:%M:%S", errors="coerce")


def scalar_type(source: Any) -> Any:
    """Scalar type of ``NDArray[T]``; None for a TypeVar (unparameterized model)."""
    args = get_args(source)
    if len(args) == 2:  # older numpy: NDArray[T] is ndarray[shape, dtype[T]]
        args = get_args(args[1])
    return None if isinstance(args[0], TypeVar) else args[0]
