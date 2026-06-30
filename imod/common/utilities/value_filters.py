import numbers
from typing import Any

import numpy as np
import xarray as xr
from xarray.core.utils import is_scalar

from imod.typing import GridDataArray, GridDataset


def is_scalar_nan(da: GridDataArray):
    """
    Test if is_scalar_nan, carefully avoid loading grids in memory
    """
    scalar_data: bool = is_scalar(da)
    if scalar_data:
        stripped_value = enforce_scalar(da)
        return isinstance(stripped_value, numbers.Real) and np.isnan(stripped_value)  # type: ignore[call-overload]
    return False


def is_valid(value: Any) -> bool:
    """
    Filters values that are None, False, or a numpy.bool_ False.
    Needs to be this specific, since 0.0 and 0 are valid values, but are
    equal to a boolean False.
    """
    # Test singletons
    if value is False or value is None:
        return False
    # Test numpy bool (not singleton)
    elif isinstance(value, np.bool_) and not value:
        return False
    # When dumping to netCDF and reading back, None will have been
    # converted into a NaN. Only check NaN if it's a floating type to avoid
    # TypeErrors.
    elif np.issubdtype(type(value), np.floating) and np.isnan(value):
        return False
    else:
        return True


def is_empty_dataarray(da: Any) -> bool:
    return isinstance(da, xr.DataArray) and enforce_scalar(da.isnull().all())


def get_scalar_variables(ds: GridDataset) -> list[str]:
    """Returns scalar variables in a dataset."""
    return [var for var, arr in ds.variables.items() if is_scalar(arr)]


def enforce_scalar(a: GridDataArray) -> Any:
    """Enforce scalar value from array."""
    if a.size == 1:
        return a.compute().item()
    raise ValueError(f"Array has size {a.size}, expected size 1.")
