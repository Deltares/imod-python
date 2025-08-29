"""
This module contains all kinds of utilities to work with layers.
"""

from imod.typing import GridDataArray
from imod.typing.grid import preserve_gridtype


@preserve_gridtype
def get_upper_active_layer_number(active: GridDataArray) -> GridDataArray:
    """
    Returns planar grid of integers with the layer number of the lower
    most active cell.

    Parameters
    ----------
    active: {xr.DataArray, xu.UgridDataArray}
        Grid of booleans (..., layer, y, x) designating active cell.

    Returns
    -------
    upper_active_layer: {xr.DataArray, xu.UgridDataArray}
        planar grid of integers (..., y, x) with the layer number of the uppermost
        active cell.

    Examples
    --------
    To get the layer numbers of the uppermost active cells for an idomain grid:

    >>> active = idomain > 0
    >>> upper_active_layer = get_upper_active_layer_number(active)

    To get the layer numbers of the uppermost active cells for a data grid with
    floats, where ``np.nan`` indicates inactive cells:

    >>> active = data.notnull()
    >>> upper_active_layer = get_upper_active_layer_number(active)
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
    active: {xr.DataArray, xu.UgridDataArray}
        Grid of booleans (..., layer, y, x) designating active cell.
    
    Returns
    -------
    upper_active_grid_cells: {xr.DataArray, xu.UgridDataArray}
        Grid of booleans (..., layer, y, x) designating location of the uppermost
        active grid cell.

    Examples
    --------
    To get the uppermost active grid cells of an idomain grid and mask idomain
    with it:

    >>> active = idomain > 0
    >>> upper_active = get_upper_active_grid_cells(active)
    >>> idomain_upper = idomain.where(upper_active, 0)

    To get the uppermost active grid cells of data grid with floats, where
    ``np.nan`` indicates inactive cells and use it to mask data:

    >>> active = data.notnull()
    >>> upper_active = get_upper_active_grid_cells(active)
    >>> data_upper = data.where(upper_active)
    """
    layer = active.coords["layer"]
    upper_active_layer = get_upper_active_layer_number(active)
    return layer == upper_active_layer


@preserve_gridtype
def get_lower_active_layer_number(active: GridDataArray) -> GridDataArray:
    """
    Returns planar grid of integers with the layer number of the lower
    most active cell.

    Parameters
    ----------
    active: {xr.DataArray, xu.UgridDataArray}
        Grid of booleans (..., layer, y, x) designating active cell.

    Returns
    -------
    lower_active_layer: {xr.DataArray, xu.UgridDataArray}
        Planar grid of integers (..., y, x) with the layer number of the lowermost
        active cell.

    Examples
    --------
    To get the layer numbers of the lowermost active cells for an idomain grid:

    >>> active = idomain > 0
    >>> lower_active_layer = get_lower_active_layer_number(active)

    To get the layer numbers of the lowermost active cells for a data grid with
    floats, where ``np.nan`` indicates inactive cells:

    >>> active = data.notnull()
    >>> lower_active_layer = get_lower_active_layer_number(active)
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
    active: {xr.DataArray, xu.UgridDataArray}
        Grid of booleans (..., layer, y, x) designating active cell.
    
    Returns
    -------
    lower_active_grid_cells: {xr.DataArray, xu.UgridDataArray}
        Grid of booleans (..., layer, y, x) designating location of the
        lowermost active grid cell.

    Examples
    --------
    To get the lowermost active grid cells of an idomain grid:

    >>> active = idomain > 0
    >>> lower_active = get_lower_active_grid_cells(active)
    >>> idomain_lower = idomain.where(lower_active, 0)
    
    To get the lowermost active grid cells of data grid with floats, where
    ``np.nan`` indicates inactive cells:

    >>> active = data.notnull()
    >>> lower_active = get_lower_active_grid_cells(active)
    >>> data_lower = data.where(lower_active)
    """
    layer = active.coords["layer"]
    lower_active_layer = get_lower_active_layer_number(active)
    return layer == lower_active_layer
