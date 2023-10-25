from typing import Any

import numpy as np
import xarray as xr
from xarray.core.utils import is_scalar

from imod.typing.grid import GridDataArray


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


def is_dataarray_none(datarray: Any) -> bool:
    return isinstance(datarray, xr.DataArray) and datarray.isnull().values.all()


def get_scalar_variables(ds: GridDataArray) -> list[str]:
    """Returns scalar variables in a dataset."""
    return [var for var, arr in ds.variables.items() if is_scalar(arr)]
