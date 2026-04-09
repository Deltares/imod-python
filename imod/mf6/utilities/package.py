from datetime import datetime
from typing import Optional

import numpy as np
import xarray as xr

import imod
from imod.common.interfaces.ipackage import IPackage
from imod.util.expand_repetitions import expand_repetitions


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


def set_repeat_stress_if_available(
    repeat: Optional[list[datetime]],
    time_min: datetime,
    time_max: datetime,
    optional_package: Optional[IPackage],
) -> None:
    """Set repeat stress for optional package if repeat is not None."""
    if repeat is not None:
        if optional_package is not None:
            times = expand_repetitions(repeat, time_min, time_max)
            optional_package.dataset["repeat_stress"] = get_repeat_stress(times)
