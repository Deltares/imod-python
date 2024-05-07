from typing import Union

import numpy as np
import xarray as xr

from imod.typing.grid import full_like, zeros_like


def convert_ibound_to_idomain(
    ibound: xr.DataArray, thickness: xr.DataArray
) -> xr.DataArray:
    # Convert IBOUND to IDOMAIN
    # -1 to 1, these will have to be filled with
    # CHD cells.
    idomain = np.abs(ibound)

    # Thickness <= 0 -> IDOMAIN = -1
    active_and_zero_thickness = (thickness <= 0) & (idomain == 1)
    # Don't make cells at top or bottom vpt, these should be inactive.
    # First, set all potential vpts to nan to be able to utilize ffill and bfill
    idomain_float = idomain.where(~active_and_zero_thickness)
    passthrough = (idomain_float.ffill("layer") == 1) & (
        idomain_float.bfill("layer") == 1
    )
    # Then fill nans where passthrough with -1
    idomain_float = idomain_float.combine_first(
        full_like(idomain_float, -1.0, dtype=float).where(passthrough)
    )
    # Fill the remaining nans at tops and bottoms with 0
    return idomain_float.fillna(0).astype(int)


def fill_missing_layers(
    source: xr.DataArray, full: xr.DataArray, fillvalue: Union[float | int]
) -> xr.DataArray:
    """
    This function takes a source grid in which the layer dimension is
    incomplete. It creates a result-grid which has the same layers as the "full"
    grid, which is assumed to have all layers. The result has the values in the
    source for the layers that are in the source. For the other layers, the
    fillvalue is assigned.
    """
    layer = full.coords["layer"]
    return source.reindex(layer=layer, fill_value=fillvalue)
