"""
Utilities for reading MODFLOW6 input files.
"""
import inspect
import warnings
from collections import defaultdict
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Tuple

import dask.array
import numpy as np

from .common import (
    advance_to_header,
    advance_to_period,
    parse_dimension,
    parse_option,
    read_iterable_block,
    read_key_value_block,
)
from .grid_data import read_griddata, shape_to_max_rows
from .list_input import read_listinput


def parse_model(stripped: str, fname: str) -> Tuple[str, str, str]:
    separated = stripped.split()
    nwords = len(separated)
    if nwords == 3:
        return separated
    else:
        raise ValueError(
            "ftype, fname and pname expected. Received instead: "
            f"{','.join(separated)} in file {fname}"
        )


def parse_exchange(stripped: str, fname: str) -> Tuple[str, str, str, str]:
    separated = stripped.split()
    nwords = len(separated)
    if nwords == 4:
        return separated
    else:
        raise ValueError(
            "exgtype, exgfile, exgmnamea, exgmnameb expected. Received instead: "
            f"{','.join(separated)} in file {fname}"
        )


def parse_solutiongroup(stripped: str, fname: str) -> Tuple[str, str]:
    separated = stripped.split()
    nwords = len(separated)
    if nwords > 2:
        return separated[0], separated[1], separated[2:]
    else:
        raise ValueError(
            "Expected at least three entries: slntype, slnfname, and one model name. "
            f"Received instead: {','.join(separated)} in file {fname}"
        )


def parse_package(stripped: str, fname: str) -> Tuple[str, str, str]:
    separated = stripped.split()
    nwords = len(separated)
    if nwords == 2:
        ftype, fname = separated
        pname = ftype
    elif nwords == 3:
        ftype, fname, pname = separated
    else:
        raise ValueError(
            "Expected ftype, fname, and optionally pname. "
            f"Received instead: {','.join(separated)} in file {fname}"
        )
    return ftype, fname, pname


def parse_tdis_perioddata(stripped: str, fname: str) -> Tuple[float, int, float]:
    separated = stripped.split()
    nwords = len(separated)
    if nwords == 3:
        return float(separated[0]), int(separated[1]), float(separated[2])
    else:
        raise ValueError(
            "perlen, nstp, tsmult expected. Received instead: "
            f"{','.join(separated)} in file {fname}"
        )


def read_tdis(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        advance_to_header(f, "options")
        content = read_key_value_block(f, parse_option)
        advance_to_header(f, "dimensions")
        dimensions = read_key_value_block(f, parse_dimension)
        advance_to_header(f, "perioddata")
        content["perioddata"] = read_iterable_block(f, parse_tdis_perioddata)
    return {**content, **dimensions}


def read_dis_blockfile(path: Path, simroot: Path) -> Dict[str, Any]:
    def delr_max_rows(shape, layered) -> Tuple[int, int]:
        if layered:
            raise ValueError(f"DELR section in {path} is LAYERED")
        return 1, shape[2]

    def delc_max_rows(shape, layered) -> int:
        if layered:
            raise ValueError(f"DELC section in {path} is LAYERED")
        return 1, shape[1]

    sections = {
        "delr": (np.float64, delr_max_rows),
        "delc": (np.float64, delc_max_rows),
        "top": (np.float64, shape_to_max_rows),
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
    # https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
    TIME_UNITS = {
        "unknown": None,
        "seconds": "s",
        "minutes": "m",
        "hours": "h",
        "days": "D",
        "years": "Y",
    }
    unit = TIME_UNITS.get(tdis["units"])

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
        possibilities = ",".join(list(TIME_UNITS.keys())[1:])
        warnings.warn(
            "Cannot convert time coordinate to datetime. "
            "Falling back to (unitless) floating point time coordinate. "
            f"time_units should be one of: {possibilities}; "
            "start_date_time should be according to ISO 8601."
        )
        times = np.full(timedelta.size, 0.0)
        times[1:] += cumulative_length[:-1]

    return times


def read_solver(path: Path) -> Dict[str, str]:
    with open(path, "r") as f:
        advance_to_header(f, "options")
        options = read_key_value_block(f, parse_option)
        advance_to_header(f, "nonlinear")
        nonlinear = read_key_value_block(f, parse_option)
        advance_to_header(f, "linear")
        linear = read_key_value_block(f, parse_option)
    return {**options, **nonlinear, **linear}


def read_sim_namefile(path: Path) -> Dict[str, str]:
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
    with open(path, "r") as f:
        advance_to_header(f, "options")
        content = read_key_value_block(f, parse_option)
        advance_to_header(f, "packages")
        content["packages"] = read_iterable_block(f, parse_package)
    return content


def read_package_periods(
    f: IO[str],
    simroot: Path,
    dtype: type,
    fields: Tuple[str],
    index_columns: Tuple[str],
    max_rows: int,
    shape: Tuple[int],
) -> Tuple[List[int], Dict[str, dask.array.Array]]:
    """
    Read blockfile periods section of a "standard" MODFLOW6 package: RIV, DRN,
    GHB, etc.

    Parameters
    ----------
    f: IO[str]
        File handle.
    simroot:
    """
    periods = []
    dask_lists = defaultdict(list)
    key = advance_to_period(f)

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
        )

        # Group them by field (e.g. cond, head, etc.)
        for field, values in zip(fields, variable_values):
            dask_lists[field].append(values)

        # Store number and advance to next period
        periods.append(key)
        key = advance_to_period(f)

    # Create a dictionary of arrays
    variable_values = {
        field: dask.array.stack(dask_list, axis=0)
        for field, dask_list in dask_lists.items()
    }
    return periods, variable_values


def read_blockfile(
    path: Path,
    simroot: Path,
    fields: Tuple[str],
    shape: Tuple[int],
    dims: Tuple[str],
) -> Dict[str, Any]:
    """
    Read blockfile of a "standard" MODFLOW6 package: RIV, DRN, GHB, etc.
    """
    ndim = len(dims)
    index_columns = {
        1: ("node",),
        2: ("layer", "cell2d"),
        3: ("layer", "row", "column"),
    }.get(ndim)
    if index_columns is None:
        raise ValueError(f"Length of dimensions should be 1, 2, or 3. Received: {ndim}")

    dtype = np.dtype(
        [(column, np.int32) for column in index_columns]
        + [(field, np.float64) for field in fields]
    )

    with open(path, "r") as f:
        advance_to_header(f, "options")
        content = read_key_value_block(f, parse_option)
        advance_to_header(f, "dimensions")
        dimensions = read_key_value_block(f, parse_dimension)
        content["period_index"], content["period_data"] = read_package_periods(
            f=f,
            simroot=simroot,
            dtype=dtype,
            fields=fields,
            index_columns=index_columns,
            max_rows=dimensions["maxbound"],
            shape=shape,
        )

    return content


def get_function_kwargnames(f: Callable) -> List[str]:
    return inspect.getfullargspec(f).args
