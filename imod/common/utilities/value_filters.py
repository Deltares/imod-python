from typing import Any

import numpy as np
import xarray as xr
from xarray.core.utils import is_scalar

from imod.typing import GridDataset


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
    return isinstance(da, xr.DataArray) and da.isnull().all().item()


def get_scalar_variables(ds: GridDataset) -> list[str]:
    """Returns scalar variables in a dataset."""
    return [var for var, arr in ds.variables.items() if is_scalar(arr)]
