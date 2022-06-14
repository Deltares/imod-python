"""
Utilities for reading MODFLOW6 input files.
"""
import warnings
from collections import defaultdict
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Tuple, Union

import dask.array
import numpy as np

from .common import (
    advance_to_header,
    advance_to_period,
    parse_dimension,
    parse_option,
    read_iterable_block,
    read_key_value_block,
    split_line,
    to_float,
)
from .grid_data import read_griddata, shape_to_max_rows
from .list_input import read_listinput


def parse_model(stripped: str, fname: str) -> Tuple[str, str, str]:
    """Parse model entry in the simulation name file."""
    separated = split_line(stripped)
    nwords = len(separated)
    if nwords == 3:
        return separated
    else:
        raise ValueError(
            "ftype, fname and pname expected. Received instead: "
            f"{','.join(separated)} in file {fname}"
        )


def parse_exchange(stripped: str, fname: str) -> Tuple[str, str, str, str]:
    """Parse exchange entry in the simulation name file."""
    separated = split_line(stripped)
    nwords = len(separated)
    if nwords == 4:
        return separated
    else:
        raise ValueError(
            "exgtype, exgfile, exgmnamea, exgmnameb expected. Received instead: "
            f"{','.join(separated)} in file {fname}"
        )


def parse_solutiongroup(stripped: str, fname: str) -> Tuple[str, str]:
    """Parse solution group entry iyn the simulation name file."""
    separated = split_line(stripped)
    if "mxiter" in stripped:
        return separated

    nwords = len(separated)
    if nwords > 2:
        return separated[0], separated[1], separated[2:]
    else:
        raise ValueError(
            "Expected at least three entries: slntype, slnfname, and one model name. "
            f"Received instead: {','.join(separated)} in file {fname}"
        )


def parse_package(stripped: str, fname: str) -> Tuple[str, str, str]:
    """Parse package entry in model name file."""
    separated = split_line(stripped)
    nwords = len(separated)
    if nwords == 2:
        ftype, fname = separated
        pname = ftype[:-1]  # split the number, e.g. riv6 -> riv
    elif nwords == 3:
        ftype, fname, pname = separated
    else:
        raise ValueError(
            "Expected ftype, fname, and optionally pname. "
            f"Received instead: {','.join(separated)} in file {fname}"
        )
    return ftype, fname, pname


def parse_tdis_perioddata(stripped: str, fname: str) -> Tuple[float, int, float]:
    """Parse a single period data entry in the time discretization file."""
    separated = split_line(stripped)
    nwords = len(separated)
    if nwords >= 3:
        return to_float(separated[0]), int(separated[1]), to_float(separated[2])
    else:
        raise ValueError(
            "perlen, nstp, tsmult expected. Received instead: "
            f"{','.join(separated)} in file {fname}"
        )


def read_tdis(path: Path) -> Dict[str, Any]:
    """Read and parse the content of the time discretization file."""
    with open(path, "r") as f:
        advance_to_header(f, "options")
        content = read_key_value_block(f, parse_option)
        advance_to_header(f, "dimensions")
        dimensions = read_key_value_block(f, parse_dimension)
        advance_to_header(f, "perioddata")
        content["perioddata"] = read_iterable_block(f, parse_tdis_perioddata)
    return {**content, **dimensions}


def read_dis_blockfile(path: Path, simroot: Path) -> Dict[str, Any]:
    """Read and parse structured discretization file."""
    ROW = 1
    COLUMN = 2

    def delr_max_rows(shape, layered) -> Tuple[int, Tuple[int]]:
        if layered:
            raise ValueError(f"DELR section in {path} is LAYERED")
        return 1, (shape[COLUMN],)

    def delc_max_rows(shape, layered) -> Tuple[int, Tuple[int]]:
        if layered:
            raise ValueError(f"DELC section in {path} is LAYERED")
        return 1, (shape[ROW],)

    def top_max_rows(shape, layered) -> Tuple[int, Tuple[int]]:
        if layered:
            raise ValueError(f"TOP section in {path} is LAYERED")
        _, nrow, ncolumn = shape
        return nrow, (nrow, ncolumn)

    sections = {
        "delr": (np.float64, delr_max_rows),
        "delc": (np.float64, delc_max_rows),
        "top": (np.float64, top_max_rows),
        "botm": (np.float64, shape_to_max_rows),
        "idomain": (np.int32, shape_to_max_rows),
    }

    with open(path, "r") as f:
        advance_to_header(f, "options")
        content = read_key_value_block(f, parse_option)
        advance_to_header(f, "dimensions")
        dimensions = read_key_value_block(f, parse_dimension)
        shape = (dimensions["nlay"], dimensions["nrow"], dimensions["ncol"])
        advance_to_header(f, "griddata")
        content["griddata"] = read_griddata(f, simroot, sections, shape)

    return {**content, **dimensions}


def tdis_time(tdis: Dict[str, Any]) -> np.ndarray:
    """
    Use start_date, time_units, and period duration to create datetime
    timestaps for the stress periods.
    """
    # https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
    TIME_UNITS = {
        "unknown": None,
        "seconds": "s",
        "minutes": "m",
        "hours": "h",
        "days": "D",
        "years": "Y",
    }
    unit = TIME_UNITS.get(tdis["time_units"])

    start = None
    if "start_date_time" in tdis:
        try:
            start = np.datetime64(tdis["start_date_time"])
        except ValueError:
            pass

    cumulative_length = np.cumsum([entry[0] for entry in tdis["perioddata"]])
    if unit is not None and start is not None:
        timedelta = np.timedelta64(cumulative_length, unit)
        times = np.full(timedelta.size, start)
        times[1:] += timedelta[:-1]
    else:
        possibilities = ", ".join(list(TIME_UNITS.keys())[1:])
        warnings.warn(
            "Cannot convert time coordinate to datetime. "
            "Falling back to (unitless) floating point time coordinate. \n"
            f"time_units should be one of: {possibilities}.\n"
            "start_date_time should be according to ISO 8601."
        )
        times = np.full(cumulative_length.size, 0.0)
        times[1:] += cumulative_length[:-1]

    return times


def read_solver(path: Path) -> Dict[str, str]:
    """Read and parse content of solver config file (IMS)."""
    with open(path, "r") as f:
        advance_to_header(f, "options")
        options = read_key_value_block(f, parse_option)
        advance_to_header(f, "nonlinear")
        nonlinear = read_key_value_block(f, parse_option)
        advance_to_header(f, "linear")
        linear = read_key_value_block(f, parse_option)
    return {**options, **nonlinear, **linear}


def read_sim_namefile(path: Path) -> Dict[str, str]:
    """Read and parse content of simulation name file."""
    with open(path, "r") as f:
        advance_to_header(f, "options")
        content = read_key_value_block(f, parse_option)
        advance_to_header(f, "timing")
        timing = read_key_value_block(f, parse_option)
        advance_to_header(f, "models")
        content["models"] = read_iterable_block(f, parse_model)
        advance_to_header(f, "exchanges")
        content["exchanges"] = read_iterable_block(f, parse_exchange)
        advance_to_header(f, "solutiongroup 1")
        content["solutiongroup 1"] = read_iterable_block(f, parse_solutiongroup)
    return {**content, **timing}


def read_gwf_namefile(path: Path) -> Dict[str, Any]:
    """Read and parse content of groundwater flow name file."""
    with open(path, "r") as f:
        advance_to_header(f, "options")
        content = read_key_value_block(f, parse_option)
        advance_to_header(f, "packages")
        content["packages"] = read_iterable_block(f, parse_package)
    return content


def read_blockfile(
    path: Path,
    simroot: Path,
    sections: Dict[str, Tuple[type, Callable]],
    shape: Tuple[int],
) -> Dict[str, Any]:
    """
    Read blockfile of a "standard" MODFLOW6 package: NPF, IC, etc.

    External files are lazily read using dask, constants are lazily allocated,
    and internal values are eagerly read and then converted to dask arrays for
    consistency.

    Parameters
    ----------
    path: Path
    simroot: Path
        Root path of the simulation, used for reading external files.
    sections: Dict[str, Tuple[type, Callable]]
        Contains for every array entry its type, and a function to compute from
        shape the number of rows to read in case of internal values.
    shape: Tuple[int]
        DIS: 3-sized, DISV: 2-sized, DISU: 1-sized.

    Returns
    -------
    content: Dict[str, Any]
        Content of the block file. Grid data arrays are stored as dask arrays.
    """
    with open(path, "r") as f:
        advance_to_header(f, "options")
        content = read_key_value_block(f, parse_option)
        advance_to_header(f, "griddata")
        content["griddata"] = read_griddata(f, simroot, sections, shape)

    return content


def read_package_periods(
    f: IO[str],
    simroot: Path,
    dtype: Union[type, np.dtype],
    index_columns: Tuple[str],
    fields: Tuple[str],
    max_rows: int,
    shape: Tuple[int],
    sparse_to_dense: bool = True,
) -> Tuple[List[int], Dict[str, dask.array.Array]]:
    """
    Read blockfile periods section of a "standard" MODFLOW6 boundary
    conditions: RIV, DRN, GHB, etc.

    External files are lazily read using dask and internal values are eagerly
    read and then converted to dask arrays for consistency.

    Parameters
    ----------
    f: IO[str]
        File handle.
    simroot:
        Root path of the simulation, used for reading external files.
    dtype: Union[type, np.dtype]
        Generally a numpy structured dtype.
    index_columns: Tuple[str]
        The index columns (np.int32) of the dtype.
    fields: Tuple[str]
        The fields (generally np.float64) of the dtype.
    max_rows: int
        Number of rows to read in case of internal values.
    shape: Tuple[int]
        DIS: 3-sized, DISV: 2-sized, DISU: 1-sized.

    Returns
    -------
    period_index: np.ndarray of integers
    variable_values: Dict[str, dask.array.Array]
    """
    periods = []
    dask_lists = defaultdict(list)
    key = advance_to_period(f)

    if sparse_to_dense:
        variables = fields
    else:
        variables = index_columns + fields

    while key is not None:
        # Read the recarrays, already separated into dense arrays.
        variable_values = read_listinput(
            f=f,
            simroot=simroot,
            dtype=dtype,
            fields=fields,
            index_columns=index_columns,
            max_rows=max_rows,
            shape=shape,
            sparse_to_dense=sparse_to_dense,
        )

        # Group them by field (e.g. cond, head, etc.)
        for var, values in zip(variables, variable_values):
            dask_lists[var].append(values)

        # Store number and advance to next period
        periods.append(key)
        key = advance_to_period(f)

    # Create a dictionary of arrays
    variable_values = {
        field: dask.array.stack(dask_list, axis=0)
        for field, dask_list in dask_lists.items()
    }
    return np.array(periods) - 1, variable_values


def read_boundary_blockfile(
    path: Path,
    simroot: Path,
    fields: Tuple[str],
    shape: Tuple[int],
    sparse_to_dense: bool = True,
) -> Dict[str, Any]:
    """
    Read blockfile of a "standard" MODFLOW6 boundary condition package: RIV,
    DRN, GHB, etc.

    External files are lazily read using dask and internal values are eagerly
    read and then converted to dask arrays for consistency.

    Parameters
    ----------
    path: Path
    simroot: Path
        Root path of the simulation, used for reading external files.
    fields: Tuple[str]
        The fields (generally np.float64) of the dtype.
    shape: Tuple[int]
        DIS: 3-sized, DISV: 2-sized, DISU: 1-sized.
    sparse_to_dense: bool, default: True
        Whether to convert "sparse" COO (cell id) data into "dense" grid form
        (with implicit location). False for packages such as Well, which are
        not usually in grid form.

    Returns
    -------
    content: Dict[str, Any]
    """
    ndim = len(shape)
    index_columns = {
        1: ("node",),
        2: ("layer", "cell2d"),
        3: ("layer", "row", "column"),
    }.get(ndim)
    if index_columns is None:
        raise ValueError(f"Length of dimensions should be 1, 2, or 3. Received: {ndim}")
    dtype_fields = [(column, np.int32) for column in index_columns] + [
        (field, np.float64) for field in fields
    ]

    with open(path, "r") as f:
        advance_to_header(f, "options")
        content = read_key_value_block(f, parse_option)

        # Create the dtype from the required variables, and with potential
        # auxiliary variables.
        auxiliary = content.pop("auxiliary", None)
        if auxiliary:
            # Make sure it's iterable in case of a single value
            if isinstance(auxiliary, str):
                auxiliary = (auxiliary,)
            for aux in auxiliary:
                dtype_fields.append((aux, np.float64))

        boundnames = content.pop("boundnames", False)
        if boundnames:
            dtype_fields.append(("boundnames", str))

        # Create np.dtype, make fields and columns immutable.
        index_columns = tuple(index_columns)
        fields = tuple(fields)
        dtype = np.dtype(dtype_fields)

        advance_to_header(f, "dimensions")
        dimensions = read_key_value_block(f, parse_dimension)
        content["period_index"], content["period_data"] = read_package_periods(
            f=f,
            simroot=simroot,
            dtype=dtype,
            index_columns=index_columns,
            fields=fields,
            max_rows=dimensions["maxbound"],
            shape=shape,
            sparse_to_dense=sparse_to_dense,
        )

    return content


def read_sto_blockfile(
    path: Path,
    simroot: Path,
    sections: Dict[str, Tuple[type, Callable]],
    shape: Tuple[int],
) -> Dict[str, Any]:
    """
    Read blockfile of MODFLOW6 Storage package.

    Parameters
    ----------
    path: Path
    simroot: Path
        Root path of the simulation, used for reading external files.
    sections: Dict[str, Tuple[type, Callable]]
        Contains for every array entry its type, and a function to compute from
        shape the number of rows to read in case of internal values.
    shape: Tuple[int]
        DIS: 3-sized, DISV: 2-sized, DISU: 1-sized.

    Returns
    -------
    content: Dict[str, Any]
        Content of the block file. Grid data arrays are stored as dask arrays.
    """
    with open(path, "r") as f:
        advance_to_header(f, "options")
        content = read_key_value_block(f, parse_option)
        advance_to_header(f, "griddata")
        content["griddata"] = read_griddata(f, simroot, sections, shape)

        periods = {}
        key = advance_to_period(f)
        while key is not None:
            entry = read_key_value_block(f, parse_option)
            value = next(iter(entry.keys()))
            if value == "steady-state":
                value = False
            elif value == "transient":
                value = True
            else:
                raise ValueError(
                    f'Expected "steady-state" or "transient". Received: {value}'
                )
            periods[key] = value
            key = advance_to_period(f)

        content["periods"] = periods

    return content
