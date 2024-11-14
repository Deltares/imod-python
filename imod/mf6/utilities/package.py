from typing import Any

import numpy as np
import xarray as xr

import imod


def get_repeat_stress(times) -> xr.DataArray:
    """
    Set repeat stresses: re-use data of earlier periods.

    Parameters
    ----------
    times: Dict of datetime-like to datetime-like.
        The data of the value datetime is used for the key datetime.
    """
    keys = [
        imod.util.time.to_datetime_internal(key, use_cftime=False)
        for key in times.keys()
    ]
    values = [
        imod.util.time.to_datetime_internal(value, use_cftime=False)
        for value in times.values()
    ]
    return xr.DataArray(
        data=np.column_stack((keys, values)),
        dims=("repeat", "repeat_items"),
    )


def _is_valid(value: Any) -> bool:
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
