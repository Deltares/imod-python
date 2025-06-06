from typing import List

import numpy as np
import xarray as xr
import xugrid as xu
from plum import Dispatcher
from scipy.ndimage import binary_dilation

from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA
from imod.schemata import DTypeSchema
from imod.typing import GridDataArray

# create dispatcher instance to limit scope of typedispatching
dispatch = Dispatcher()


def _reduce_grid_except_dims(
    grid: GridDataArray, preserve_dims: List[str]
) -> GridDataArray:
    to_reduce = {dim: 0 for dim in grid.dims if dim not in preserve_dims}
    return grid.isel(**to_reduce)  # type: ignore [misc, arg-type]


def _validate_grid(grid):
    # Validate if required dimensions are present
    schemata = [BOUNDARY_DIMS_SCHEMA, DTypeSchema(np.bool_)]
    for schema in schemata:
        schema.validate(grid)


def grid_boundary_xy(grid: GridDataArray) -> GridDataArray:
    """
    Return grid boundary on the xy plane.

    Wraps the binary_dilation function.

    Parameters
    ----------
    grid : {xarray.DataArray, xugrid.UgridDataArray}
        Grid with either ``x`` and ``y`` dimensions or a face dimesion.

    Returns
    -------
    {xarray.DataArray, xugrid.UgridDataArray}
        2d grid with locations of grid boundaries
    """
    return _grid_boundary_xy(grid)


@dispatch
def _grid_boundary_xy(grid: object) -> None:
    raise TypeError(
        f"Grid should be of type DataArray or UgridDataArray, got type {type(grid)}"
    )


@dispatch  # type: ignore [no-redef]
def _grid_boundary_xy(grid: xr.DataArray) -> xr.DataArray:
    _validate_grid(grid)
    like_2d = _reduce_grid_except_dims(grid, ["x", "y"])
    boundary_grid = xr.zeros_like(like_2d)
    boundary_grid.values = binary_dilation(boundary_grid, border_value=1)
    return boundary_grid


@dispatch  # type: ignore [no-redef]
def _grid_boundary_xy(grid: xu.UgridDataArray) -> xu.UgridDataArray:
    _validate_grid(grid)
    like_2d = _reduce_grid_except_dims(grid, [grid.grid.face_dimension])
    zeros_grid = xu.zeros_like(like_2d)
    return zeros_grid.ugrid.binary_dilation(border_value=1)


def active_grid_boundary_xy(
    active: GridDataArray,
) -> GridDataArray:
    """
    Return active boundary cells on the xy plane.

    Parameters
    ----------
    active : {xarray.DataArray, xugrid.UgridDataArray}
        Grid with active cells,
        either with ``x`` and ``y`` dimensions or a face dimesion.

    Returns
    -------
    {xarray.DataArray, xugrid.UgridDataArray}
        Locations of active grid boundaries
    """

    grid_boundary = grid_boundary_xy(active)

    return active & grid_boundary
