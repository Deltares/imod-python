import numpy as np
import xarray as xr

import imod


def upper_active_layer(da, is_ibound=True, include_constant_head=False):
    """
    Function to get the upper active layer from ibound xarray.DataArray

    Parameters
    ----------
    da : xarray.DataArray
        A 3D DataArray
    is_ibound: bool, optional
        If True, ``da`` is interpreted as ibound, with values 0: inactive, 1: active, -1 constant head.
        If False, ``upper_active_layer`` is interpreted as first layer that has data.
        Default is True.
    include_constant_head : bool, optional
        If True and ``is_ibound``, also include constant head cells.
        Default is False.

    Returns
    -------
    2d xr.DataArray of layernumber of upper active model layer
    """
    da = da.load()
    if is_ibound:
        # check if indeed ibound: convertible to int
        if not da.astype(int).equals(da):
            raise ValueError(
                "Passed DataArray is no ibound, while is_bound was set to True"
            )
        # include constant head cells (?)
        if include_constant_head:
            is_active = da.fillna(0) != 0  # must be filled for argmax
        else:
            is_active = da.fillna(0) > 0
    else:
        is_active = ~da.isnull()

    # get layer of upper active cell
    da = is_active.layer.isel(layer=is_active.argmax(dim="layer"))
    da = da.drop("layer")

    # skip where no active cells
    return da.where(is_active.sum(dim="layer") > 0)
