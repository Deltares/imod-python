import typing
from typing import Dict

import numpy as np
import xarray as xr
import xugrid as xu

from imod.typing.grid import GridDataArray, zeros_like
from imod.util import spatial_reference

DomainSlice = Dict[str, slice | np.ndarray]


def get_active_domain_slice(active: GridDataArray) -> DomainSlice:
    if isinstance(active, xr.DataArray):
        grid = active.where(active > 0, drop=True)

        _, xmin, xmax, _, ymin, ymax = spatial_reference(grid)
        x_slice = slice(int(xmin), int(xmax))
        y_slice = slice(int(ymax), int(ymin))
        return {"y": y_slice, "x": x_slice}

    if isinstance(active, xu.UgridDataArray):
        active_indices = np.where(active > 0)[0]
        return {f"{active.ugrid.grid.face_dimension}": active_indices}

    raise TypeError(f"Unknown grid type {active}")


def broadcast_to_full_domain(
    idomain: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
) -> typing.Tuple[GridDataArray, GridDataArray]:
    """
    Broadcast the bottom and top array to have the same shape as the idomain
    """
    bottom = idomain * bottom
    top = (
        idomain * top
        if hasattr(top, "coords") and "layer" in top.coords
        else create_layered_top(bottom, top)
    )

    return top, bottom


def create_layered_top(bottom: GridDataArray, top: GridDataArray) -> GridDataArray:
    """
    Create a top array with layers from a single top array and a full bottom array
    """
    new_top = zeros_like(bottom)
    new_top[0] = top
    new_top[1:] = bottom[0:-1].values

    return new_top
