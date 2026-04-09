from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from imod.typing import GridDataArray


def remove_inactive(ds: xr.Dataset, active: xr.DataArray) -> xr.Dataset:
    """
    Drop list-based input cells in inactive cells.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset with list-based input. Needs "cellid" variable.
    active: xr.DataArray
        Grid with active cells.
    """

    def unstack_columns(array):
        # Unstack columns:
        # https://stackoverflow.com/questions/64097426/is-there-unstack-in-numpy
        # Make sure to use tuples, since these get the special treatment
        # which we require for the indexing:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        return tuple(np.moveaxis(array, -1, 0))

    if "cellid" not in ds.data_vars:
        raise ValueError("Missing variable 'cellid' in dataset")
    if "ncellid" not in ds.dims:
        raise ValueError("Missing dimension 'ncellid' in dataset")

    cellid_zero_based = ds["cellid"].values - 1
    cellid_indexes = unstack_columns(cellid_zero_based)
    valid = active.values[cellid_indexes].astype(bool)

    return ds.loc[{"ncellid": valid}]


def assign_datetime_coords(
    da: GridDataArray,
    simulation_start_time: np.datetime64,
    time_unit: Optional[str] = "d",
) -> GridDataArray:
    if "time" not in da.coords:
        raise ValueError(
            "cannot convert time column, because a time column could not be found"
        )

    time = pd.Timestamp(simulation_start_time) + pd.to_timedelta(
        da["time"], unit=time_unit
    )
    return da.assign_coords(time=time)


def assign_index(arg):
    if isinstance(arg, xr.DataArray):
        arg = arg.values
    elif not isinstance(arg, (np.ndarray, list, tuple)):
        raise TypeError("should be a tuple, list, or numpy array")

    arr = np.array(arg)
    if arr.ndim != 1:
        raise ValueError("should be 1D")

    return xr.DataArray(arr, dims=("index",))
