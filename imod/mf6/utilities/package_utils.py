import numpy as np
import xarray as xr

import imod
from imod.mf6.interfaces.ipackagebase import IPackageBase


def set_repeat_stress(package: IPackageBase, times) -> None:
    """
    Set repeat stresses: re-use data of earlier periods.

    Parameters
    ----------
    package : The package in which to set the repeated stress
    times: Dict of datetime-like to datetime-like.
        The data of the value datetime is used for the key datetime.
    """
    keys = [imod.wq.timeutil.to_datetime(key, use_cftime=False) for key in times.keys()]
    values = [
        imod.wq.timeutil.to_datetime(value, use_cftime=False)
        for value in times.values()
    ]
    package.dataset["repeat_stress"] = xr.DataArray(
        data=np.column_stack((keys, values)),
        dims=("repeat", "repeat_items"),
    )
