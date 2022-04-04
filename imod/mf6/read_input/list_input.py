"""
Utilities for reading MODFLOW6 list input.
"""
from pathlib import Path
from typing import IO, List, Tuple

import dask.array
import numpy as np

from .common import (
    read_external_binaryfile,
    read_external_textfile,
    read_internal,
    strip_line,
)


def recarr_to_dense(recarr, index_columns, fields, shape) -> List[np.ndarray]:
    """
    Convert a record array to separate numpy arrays. Uses the index columns to
    place the values in a dense array form.
    """
    # MODFLOW6 is 1-based, Python is 0-based
    indices = [recarr[column] - 1 for column in index_columns]
    variables = []
    for field in fields:
        data = np.full(shape, np.nan)
        data[indices] = recarr[field]
        variables.append(data)
    return variables


def read_internal_listinput(
    f: IO[str],
    dtype: type,
    index_columns: Tuple[str],
    fields: Tuple[str],
    shape: Tuple[int],
    max_rows: int,
) -> List[dask.array.Array]:
    recarr = read_internal(f, max_rows, dtype)
    return recarr_to_dense(recarr, index_columns, fields, shape)


def read_external_listinput(
    path: Path,
    dtype: type,
    index_columns: Tuple[str],
    fields: Tuple[str],
    shape: Tuple[int],
    binary: bool,
):
    """
    Read external list input, separate and reshape to a dense array form.
    """
    if binary:
        recarr = read_external_binaryfile(path, dtype)
    else:
        recarr = read_external_textfile(path, dtype)
    return recarr_to_dense(recarr, index_columns, fields, shape)


def read_listinput(
    f: IO[str],
    simroot: Path,
    dtype: type,
    index_columns: Tuple[str],
    fields: Tuple[str],
    shape: Tuple[int],
    max_rows: int,
) -> List[dask.array.Array]:
    """
    MODFLOW6 list input reading functionality.

    MODFLOW6 list input is "sparse": it consists of a cell id and a number of
    values. Depending on whether the model is discretized according to DIS,
    DISV, or DISU; this cell id may be a tuple of size 3, 2, or 1.

    Parameters
    ----------
    f: IO[str]
        File handle.
    simroot: Path
        Root path of simulation. Used for reading external files.
    dtype: type

    index_columns: Tuple[str]
    fields: Tuple[str]
    shape: Tuple[int]
    max_rows: int

    """
    # Store position so week can move back in the file if data is stored
    # internally.
    position = f.tell()

    # Read and check the first line.
    firstline = f.readline()
    stripped = strip_line(firstline)
    separated = stripped.split()
    first = separated[0]

    if first == "open/close":
        fname = separated[1]
        binary = "(binary)" in stripped
        path = simroot / fname

        nout = len(fields)
        x = dask.delayed(read_external_listinput, nout=nout)(
            path, dtype, index_columns, fields, shape, binary
        )
        variable_values = [
            dask.array.from_delayed(a, shape=shape, dtype=dtype) for a in x
        ]
    else:
        f.seek(position)
        x = read_internal_listinput(f, dtype, index_columns, fields, shape, max_rows)
        variable_values = [dask.array.from_array(a) for a in x]

    return variable_values
