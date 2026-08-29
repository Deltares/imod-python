import os
import struct
from typing import BinaryIO, Optional

import dask
import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.out.common import FilePath, _to_nan
from imod.mf6.utilities.dataset import assign_datetime_coords

LEN_RECORD1 = 100
LEN_RECORD2 = 4
TIMESTAMP_SIZE = 8


def read_times(
    f: BinaryIO, ntime: int, nobs: int
) -> np.ndarray:
    times = np.empty(ntime, dtype=np.float64)
    for i in range(ntime):
        times[i] = struct.unpack("d", f.read(8))[0]
        f.seek(nobs * 8, 1)
    return times


def read_timestep(
    path: FilePath, dry_nan: bool, nobs: int, pos: int,
) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(pos)
        a = np.fromfile(f, np.float64, nobs)
    return _to_nan(a, dry_nan)


def read_record1(f: BinaryIO) -> int:
    recordtype = f.read(5).decode("utf-8").strip()
    precision = f.read(6).decode("utf-8").strip()
    lenobsname = f.read(4).decode("utf-8").strip()
    blanks = f.read(85).decode("utf-8").strip()
    if not recordtype == "cont":
        raise ValueError(f'recordtype is not "cont", but: "{recordtype}"')
    if not precision == "double":
        raise ValueError(f'precision is not "double", but: "{recordtype}"')
    if not blanks == "":
        raise ValueError(f'Expected 85 blanks, but received: "{blanks}"')
    return int(lenobsname.strip())


def read_record2(f: BinaryIO) -> int:
    return struct.unpack("i", f.read(4))[0]


def read_record3(f: BinaryIO, len_obsname: int, nobs: int) -> np.ndarray:
    rawnames = np.fromfile(
        file=f,
        dtype=f"<S{len_obsname}",
        count=nobs,
    )
    return np.char.strip(rawnames).astype(str)


def open_obs_bsv(
    path: FilePath,
    dry_nan: bool=False,
    simulation_start_time: Optional[np.datetime64] = None,
    time_unit: Optional[str] = "d"
) -> xr.DataArray:
    """
    Open modflow6 observation package binary simulated values ("bsv") file.

    The data is lazily read per timestep.

    Parameters
    ----------
    path: Union[str, pathlib.Path]
    dry_nan: bool, default value: False.
        Whether to convert dry values to NaN.
    simulation_start_time : Optional datetime
        The time and date correpsonding to the beginning of the simulation.
        Use this to convert the time coordinates of the output array to
        calendar time/dates. time_unit must also be present if this argument is present.
    time_unit: Optional str
        The time unit MF6 is working in, in string representation.
        Only used if simulation_start_time was provided.
        Admissible values are:
        ns -> nanosecond
        ms -> microsecond
        s -> second
        m -> minute
        h -> hour
        d -> day
        w -> week
        Units "month" or "year" are not supported, as they do not represent unambiguous timedelta values durations.

    Returns
    -------
    observations: xr.DataArray
    """
    filesize = os.path.getsize(path)
    with open(path, "rb") as f:
        len_obsname = read_record1(f)
        nobs = read_record2(f)
        obsnames = read_record3(f, len_obsname, nobs)
        len_record3 = len_obsname * nobs
        # For a timestep, the timestamp is stored and all observations, as doubles.
        len_timestep = (nobs + 1) * 8
        ntime = (filesize - (LEN_RECORD1 + LEN_RECORD2 + len_record3)) // len_timestep
        times = read_times(f, ntime, nobs)

    dask_list = []
    initial_skip = LEN_RECORD1 + LEN_RECORD2 + len_record3 + TIMESTAMP_SIZE
    for i in range(ntime):
        pos = initial_skip + (i * len_timestep)
        a = dask.delayed(read_timestep)(path, dry_nan, nobs, pos)
        x = dask.array.from_delayed(a, shape=(nobs,), dtype=np.float64)
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    data_array = xr.DataArray(
        data=daskarr,
        dims=("time", "observation"),
        coords={"time": times, "observation": obsnames}
    )
    if simulation_start_time is not None:
        data_array = assign_datetime_coords(
            data_array, simulation_start_time, time_unit
        )
    return data_array


def read_obs_csv(
    path: FilePath,
    dry_nan: bool=False,
    simulation_start_time: Optional[np.datetime64] = None,
    time_unit: Optional[str] = "d"
) -> xr.DataArray:
    """
    Read modflow6 observation package CSV file.

    Unlike the ``open_obs_bsv`` function, data is directly read into memory.

    Parameters
    ----------
    path: Union[str, pathlib.Path]
    dry_nan: bool, default value: False.
        Whether to convert dry values to NaN.
    simulation_start_time : Optional datetime
        The time and date correpsonding to the beginning of the simulation.
        Use this to convert the time coordinates of the output array to
        calendar time/dates. time_unit must also be present if this argument is present.
    time_unit: Optional str
        The time unit MF6 is working in, in string representation.
        Only used if simulation_start_time was provided.
        Admissible values are:
        ns -> nanosecond
        ms -> microsecond
        s -> second
        m -> minute
        h -> hour
        d -> day
        w -> week
        Units "month" or "year" are not supported, as they do not represent unambiguous timedelta values durations.

    Returns
    -------
    observations: xr.DataArray
    """

    df = pd.read_csv(path, index_col=0)
    times = df.index
    obsnames = df.columns
    data = _to_nan(df.to_numpy(), dry_nan)
    data_array = xr.DataArray(
        data=data,
        dims=("time", "observation"),
        coords={"time": times, "observation": obsnames}
    )
    if simulation_start_time is not None:
        data_array = assign_datetime_coords(
            data_array, simulation_start_time, time_unit
        )
    return data_array
