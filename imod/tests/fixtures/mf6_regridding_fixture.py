import numpy as np
import xarray as xr
import xugrid as xu

def grid_data_structured(dtype, value, cellsize) -> xr.DataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size.
    """
    horizontal_range = 10
    y = np.arange(horizontal_range, -cellsize, -cellsize)
    x = np.arange(0, horizontal_range + cellsize, cellsize)

    nlayer = 3

    shape = nlayer, len(x), len(y)
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlayer + 1)

    coords = {"layer": layer, "y": y, "x": x, "dx": cellsize, "dy": cellsize}

    da = xr.DataArray(np.ones(shape, dtype=dtype) * value, coords=coords, dims=dims)

    return da

def grid_data_structured_layered(dtype, value, cellsize) -> xr.DataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size. The values are
    multiplied with the layer index.
    """
    horizontal_range = 10
    y = np.arange(horizontal_range, -cellsize, -cellsize)
    x = np.arange(0, horizontal_range + cellsize, cellsize)

    nlayer = 3

    shape = nlayer, len(x), len(y)
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlayer + 1)

    coords = {"layer": layer, "y": y, "x": x, "dx": cellsize, "dy": cellsize}

    da = xr.DataArray(np.ones(shape, dtype=dtype), coords=coords, dims=dims)
    for ilayer in range(1, nlayer + 1):
        layer_value = ilayer * value
        da.loc[dict(layer=ilayer)] = layer_value
    return da


def grid_data_unstructured(dtype, value, cellsize) -> xu.UgridDataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size.
    First a regular grid is constructed and then this is converted to an ugrid dataarray.
    """
    return xu.UgridDataArray.from_structured(
        grid_data_structured(dtype, value, cellsize)
    )


def grid_data_unstructured_layered(dtype, value, cellsize) -> xu.UgridDataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size. The values are
    multiplied with the layer index. First a regular grid is constructed and then this is converted to an ugrid dataarray.
    """
    return xu.UgridDataArray.from_structured(
        grid_data_structured_layered(dtype, value, cellsize)
    )
