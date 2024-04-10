"""
This module contains all kinds of utilities to work with layers.
"""

from imod.typing import GridDataArray
from imod.typing.grid import preserve_gridtype, zeros_like


@preserve_gridtype
def get_upper_active_layer_number(active: GridDataArray) -> GridDataArray:
    """
    Returns planar grid of integers with the layer number of the upper most
    active cell.

    Parameters
    ----------
    active: {DataArray, UgridDataArray}
        Grid of booleans designating active cell.
    """
    layer = active.coords["layer"]
    # Set nodata to sentinel value to prevent a dtype shift from integer to
    # float as np.nan forces float.
    nodata = layer.max() + 1
    return layer.where(active, nodata).min("layer")


@preserve_gridtype
def get_upper_active_grid_cells(active: GridDataArray) -> GridDataArray:
    """
    Returns grid of booleans designating location of the uppermost active grid
    cell.

    Parameters
    ----------
    active: {DataArray, UgridDataArray}
        Grid of booleans designating active cell.
    """
    layer = active.coords["layer"]
    upper_active_layer = get_upper_active_layer_number(active)
    return layer == upper_active_layer


@preserve_gridtype
def get_lower_active_layer_number(active: GridDataArray) -> GridDataArray:
    """
    Returns two-dimensional grid of integers with the layer number of the lower
    most active cell.

    Parameters
    ----------
    active: {DataArray, UgridDataArray}
        Grid of booleans designating active cell.
    """
    layer = active.coords["layer"]
    # Set nodata to sentinel value to prevent a dtype shift from integer to
    # float as np.nan forces float.
    nodata = layer.min() - 1
    return layer.where(active, nodata).max("layer")


@preserve_gridtype
def get_lower_active_grid_cells(active: GridDataArray) -> GridDataArray:
    """
    Returns grid of booleans designating location of the lowermost active grid
    cell.

    Parameters
    ----------
    active: {DataArray, UgridDataArray}
        Grid of booleans designating active cell.
    """
    layer = active.coords["layer"]
    lower_active_layer = get_lower_active_layer_number(active)
    return layer == lower_active_layer


def create_layered_top(bottom: GridDataArray, top: GridDataArray) -> GridDataArray:
    """
    Create a top array with a layer dimension, from a top array with no layer
    dimension and a bottom array with a layer dimension. The (output) top of
    layer n is assigned the bottom of layer n-1.

    Parameters
    ----------
    bottom: {DataArray, UgridDataArray}
        Bottoms with layer dimension
    top: {DataArray, UgridDataArray}
        Top, without layer dimension

    Returns
    -------
    new_top: {DataArray, UgridDataArray}
        Top with layer dimension.
    """
    new_top = zeros_like(bottom)
    new_top[0] = top
    new_top[1:] = bottom[0:-1].values

    return new_top
