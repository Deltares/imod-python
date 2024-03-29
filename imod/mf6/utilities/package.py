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
