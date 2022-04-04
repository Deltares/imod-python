"""
Utilities for reading MODFLOW6 GRID DATA input.
"""
from pathlib import Path
from typing import IO, Any, Callable, Dict, Tuple

import dask.array
import numpy as np

from .common import (
    end_of_file,
    find_entry,
    read_external_binaryfile,
    read_external_textfile,
    read_internal,
    strip_line,
)


def advance_to_griddata_section(f: IO[str], section: str) -> None:
    line = None
    # Empty line is end-of-file
    while not end_of_file(line):
        line = f.readline()
        stripped = strip_line(line)
        # Return if start has been found
        if stripped == section:
            return False
        elif stripped == f"{section} layered":
            return True
        # Continue if line is empty
        elif stripped == "":
            continue
        # Raise if the line has (unexpected) content
        else:
            break
    raise ValueError(f'"{section}" is not present in file {f.name}')


def shape_to_max_rows(shape: Tuple[int], layered: bool) -> int:
    ndim = len(shape)
    if ndim == 3:
        nlayer, nrow, _ = shape
        max_rows_layered = nrow
        max_rows = nlayer * nrow

    elif ndim == 2:
        nlayer, _ = shape
        max_rows_layered = 1
        max_rows = nlayer

    elif ndim == 1:
        if layered:
            raise ValueError(
                "LAYERED section detected. DISU does not support LAYERED input."
            )
        nlayer = None
        max_rows_layered = None
        max_rows = 1

    else:
        raise ValueError(
            f"length of shape should be 1, 2, or 3. Received shape of length: {ndim}"
        )

    if layered:
        return max_rows_layered, shape[1:]
    else:
        return max_rows, shape


def constant(value: Any, shape: Tuple[int], dtype: type) -> dask.array.Array:
    return dask.array.full(shape, value, dtype=dtype)


def read_internal_griddata(
    f: IO[str], dtype: type, shape: Tuple[int], max_rows: int
) -> np.ndarray:
    return read_internal(f, max_rows, dtype).reshape(shape)


def read_external_griddata(
    path: Path, dtype: type, shape: Tuple[int], binary: bool
) -> np.ndarray:
    if binary:
        a = read_external_binaryfile(path, dtype)
    else:
        a = read_external_textfile(path, dtype)
    return a.reshape(shape)


def read_array(
    f: IO[str],
    simroot: Path,
    dtype: type,
    max_rows: int,
    shape: Tuple[int],
) -> dask.array.Array:
    """
    MODFLOW6 READARRAY functionality for grid data.

    Parameters
    ----------

    Returns
    -------
    array: dask.array.Array
    """
    firstline = f.readline()
    stripped = strip_line(firstline)
    separated = stripped.split()
    first = separated[0]

    if first == "constant":
        factor = None
        array = constant(separated[1], shape, dtype)
    elif first == "internal":
        factor = find_entry(stripped, "factor", float)
        a = read_internal_griddata(f, dtype, shape, max_rows)
        array = dask.array.from_array(a)
    elif first == "open/close":
        factor = find_entry(stripped, "factor", float)
        fname = separated[1]
        binary = "(binary)" in stripped
        path = simroot / fname
        a = dask.delayed(read_external_griddata)(path, dtype, shape, binary)
        array = dask.array.from_delayed(a, shape=shape, dtype=dtype)
    else:
        raise ValueError(
            'Expected "constant", "internal" or "open/close". Received instead: '
            f"{stripped}"
        )

    if factor is not None:
        array = array * factor

    return array


def read_griddata(
    f: IO[str],
    simroot: Path,
    sections: Dict[str, Tuple[type, Callable]],
    shape: Tuple[int],
) -> Dict[str, dask.array.Array]:
    """
    Read GRID DATA section.

    Constants are lazily allocated; external files are lazily read.  Internal
    arrays are eagerly read, but converted to a dask array for consistency.

    Parameters
    ----------
    f: IO[str]
    simroot: Path
        Root path of simulation. Used for reading external files.
    sections: Dict[str, (type, Callabe)]
        Dictionary containing type of grid data entry, and function to compute
        number of max_rows to read.
    shape: Tuple[int]
        Full shape of model read from DIS file. Should have length:
            * DIS: 3
            * DISV: 2
            * DISU: 1

    Returns
    -------
    variables: Dict[str, xr.DataArray]
    """
    content = {}
    for section, (dtype, compute_max_rows) in sections.items():
        try:
            layered = advance_to_griddata_section(f, section)
            max_rows, read_shape = compute_max_rows(shape, layered)
            kwargs = {
                "f": f,
                "simroot": simroot,
                "dtype": dtype,
                "shape": read_shape,
                "max_rows": max_rows,
            }

            if layered:
                nlayer = shape[0]
                data = dask.array.stack(
                    [read_array(**kwargs) for _ in range(nlayer)],
                    axis=0,
                )
            else:
                data = read_array(**kwargs)

            content[section] = data

        except Exception as e:
            raise type(e)(f"{e}\n Error reading {section} in {f.name}") from e

    return content
