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
    split_line,
    strip_line,
    to_float,
)


def advance_to_griddata_section(f: IO[str]) -> Tuple[str, bool]:
    line = None
    # Empty line is end-of-file.
    # Should always exit early in while loop, err otherwise.
    while not end_of_file(line):
        line = f.readline()
        stripped = strip_line(line)
        if stripped == "":
            continue
        elif "end" in stripped:
            return None, False
        elif "layered" in stripped:
            layered = True
            section = split_line(stripped)[0]
            return section, layered
        else:
            layered = False
            section = stripped
            return section, layered
    raise ValueError(f"No end of griddata specified in {f.name}")


def shape_to_max_rows(shape: Tuple[int], layered: bool) -> Tuple[int, Tuple[int]]:
    """
    Compute the number of rows to read in case the data is internal. In case
    of DIS, the number of (table) rows equals the number of layers times the
    number of (model) rows; a single row then contains ncolumn values.

    A DISV does not have rows and columns, the number of rows is equal to the
    number of layers.

    In case of LAYERED, each layer is written on a separate row.

    In case of DISU, all values are written one a single row, and LAYERED input
    is not allowed.

    Parameters
    ----------
    shape: Tuple[int]
    layered: bool

    Returns
    ----------
    max_rows: int
        Reduced number if layered is True.
    read_shape: Tuple[int]
        First dimension removed if layered is True.
    """
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
    return read_internal(f=f, dtype=dtype, max_rows=max_rows).reshape(shape)


def read_external_griddata(
    path: Path, dtype: type, shape: Tuple[int], binary: bool
) -> np.ndarray:
    max_rows = np.product(shape)
    if binary:
        a = read_external_binaryfile(path, dtype, max_rows)
    else:
        a = read_external_textfile(path, dtype, max_rows)
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

    External files are lazily read using dask, constants are lazily allocated,
    and internal values are eagerly read and then converted to dask arrays for
    consistency.

    Parameters
    ----------
    f: IO[str]
    simroot: Path
    dtype: type
    max_rows: int
        Number of rows to read in case of internal data values.
    shape: Tuple[int]

    Returns
    -------
    array: dask.array.Array of size ``shape``
    """
    firstline = f.readline()
    stripped = strip_line(firstline)
    separated = split_line(stripped)
    first = separated[0]

    if first == "constant":
        factor = None
        array = constant(separated[1], shape, dtype)
    elif first == "internal":
        factor = find_entry(stripped, "factor", to_float)
        a = read_internal_griddata(f, dtype, shape, max_rows)
        array = dask.array.from_array(a)
    elif first == "open/close":
        factor = find_entry(stripped, "factor", to_float)
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

    External files are lazily read using dask, constants are lazily allocated,
    and internal values are eagerly read and then converted to dask arrays for
    consistency.

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
    variables: Dict[str, dask.array.Array]
    """
    content = {}
    try:
        section, layered = advance_to_griddata_section(f)
        while section is not None:
            try:
                dtype, compute_max_rows = sections[section]
            except KeyError:
                raise ValueError(
                    f"Unexpected section: {section}. "
                    f"Expected one of: {', '.join(sections.keys())}"
                )
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

            # Advance to next section
            section, layered = advance_to_griddata_section(f)

    except Exception as e:
        raise type(e)(f"{e}\n Error reading {f.name}") from e

    return content
