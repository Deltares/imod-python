import typing
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu

import imod
from imod.prepare.layer import create_layered_top
from imod.typing import GridDataArray
from imod.typing.grid import zeros_like
from imod.util.spatial import spatial_reference

DomainSlice = Dict[str, slice | np.ndarray]


def get_active_domain_slice(active: GridDataArray) -> DomainSlice:
    if isinstance(active, xr.DataArray):
        grid = active.where(active > 0, drop=True)

        _, xmin, xmax, _, ymin, ymax = spatial_reference(grid)
        x_slice = slice(xmin, xmax)
        y_slice = slice(ymax, ymin)
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


def to_cell_idx(idomain: xr.DataArray) -> xr.DataArray:
    """
    Assigns an unique index to each cell in the domain
    """
    index = np.arange(idomain.size).reshape(idomain.shape)
    domain_index = zeros_like(idomain)
    domain_index.values = index

    return domain_index


def create_geometric_grid_info(active: xr.DataArray) -> pd.DataFrame:
    dx = np.abs(imod.util.spatial.coord_reference(active.x)[0])
    dy = np.abs(imod.util.spatial.coord_reference(active.y)[0])

    global_cell_indices = to_cell_idx(active)
    num_y, num_x = active.shape
    dx = np.broadcast_to(np.broadcast_to(np.array(dx), (1, num_x)), (num_y, num_x))
    dy = np.broadcast_to(np.broadcast_to(np.array([dy]).T, (num_y, 1)), (num_y, num_x))

    y, x = zip(*active.stack(z=["y", "x"]).z.values)

    return pd.DataFrame(
        {
            "global_idx": global_cell_indices.values.flatten(),
            "x": x,
            "y": y,
            "dx": dx.flatten(),
            "dy": dy.flatten(),
        }
    )


def create_smallest_target_grid(*grids: xr.DataArray) -> xr.DataArray:
    """
    Create smallest target grid from multiple structured grids. This is the grid
    with smallest extent and finest resolution amongst all provided grids.
    """
    dx_ls, xmin_ls, xmax_ls, dy_ls, ymin_ls, ymax_ls = zip(
        *[imod.util.spatial.spatial_reference(grid) for grid in grids]
    )

    dx = min(dx_ls)
    xmin = max(xmin_ls)
    xmax = min(xmax_ls)
    dy = max(dy_ls)
    ymax = min(ymax_ls)
    ymin = max(ymin_ls)

    return imod.util.spatial.empty_2d(dx, xmin, xmax, dy, ymin, ymax)
