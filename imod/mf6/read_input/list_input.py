"""
Utilities for reading MODFLOW6 list input.
"""
from pathlib import Path
from typing import IO, List, Tuple

import dask.array
import numpy as np
import pandas as pd

from .common import read_external_binaryfile, split_line, strip_line


def recarr_to_dense(
    recarr: np.ndarray,
    index_columns: Tuple[str],
    fields: Tuple[str],
    shape: Tuple[int],
) -> List[np.ndarray]:
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


def read_text_listinput(
    path: Path,
    dtype: np.dtype,
    max_rows: int,
) -> np.ndarray:
    # This function is over three times faster than:
    # recarr = np.loadtxt(path, dtype, max_rows=max_rows)
    d = {key: value[0] for key, value in dtype.fields.items()}
    df = pd.read_csv(
        path,
        header=None,
        dtype=d,
        names=d.keys(),
        delim_whitespace=True,
        comment="#",
        nrows=max_rows,
    )
    return df.to_records(index=False)


def read_internal_listinput(
    f: IO[str],
    dtype: np.dtype,
    index_columns: Tuple[str],
    fields: Tuple[str],
    shape: Tuple[int],
    max_rows: int,
) -> List[np.ndarray]:
    # recarr = read_internal(f, max_rows, dtype)
    recarr = read_text_listinput(f, dtype, max_rows)
    return recarr_to_dense(recarr, index_columns, fields, shape)


def read_external_listinput(
    path: Path,
    dtype: np.dtype,
    index_columns: Tuple[str],
    fields: Tuple[str],
    shape: Tuple[int],
    binary: bool,
    max_rows: int,
) -> List[np.ndarray]:
    """
    Read external list input, separate and reshape to a dense array form.
    """
    if binary:
        recarr = read_external_binaryfile(path, dtype, max_rows)
    else:
        recarr = read_text_listinput(path, dtype, max_rows)
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
    dtype: np.dtype
    index_columns: Tuple[str]
    fields: Tuple[str]
    shape: Tuple[int]
    max_rows: int

    Returns
    -------
    variable_values: List[dask.array.Array]
        A dask array for every entry in ``fields``.
    """
    # Store position so week can move back in the file if data is stored
    # internally.
    position = f.tell()

    # Read and check the first line.
    firstline = f.readline()
    stripped = strip_line(firstline)
    separated = split_line(stripped)
    first = separated[0]

    if first == "open/close":
        fname = separated[1]
        path = simroot / fname
        binary = "(binary)" in stripped

        if binary and "boundnames" in dtype.fields:
            raise ValueError("(BINARY) does not support BOUNDNAMES")

        nout = len(fields)
        x = dask.delayed(read_external_listinput, nout=nout)(
            path, dtype, index_columns, fields, shape, binary, max_rows
        )
        variable_values = [
            dask.array.from_delayed(a, shape=shape, dtype=dtype) for a in x
        ]
    else:
        # Move file position back one line, and try reading internal values.
        f.seek(position)
        x = read_internal_listinput(f, dtype, index_columns, fields, shape, max_rows)
        variable_values = [dask.array.from_array(a) for a in x]

    return variable_values
