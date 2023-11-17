import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.prepare import polygonize
from imod.typing import GridDataArray, GridDataset


@typedispatch
def zeros_like(grid: xr.DataArray, *args, **kwargs):
    return xr.zeros_like(grid, *args, **kwargs)


@typedispatch
def zeros_like(grid: xu.UgridDataArray, *args, **kwargs):
    return xu.zeros_like(grid, *args, **kwargs)


@typedispatch
def ones_like(grid: xr.DataArray, *args, **kwargs):
    return xr.ones_like(grid, *args, **kwargs)


@typedispatch
def ones_like(grid: xu.UgridDataArray, *args, **kwargs):
    return xu.ones_like(grid, *args, **kwargs)


@typedispatch
def nan_like(grid: xr.DataArray, *args, **kwargs):
    return xr.full_like(grid, fill_value=np.nan, dtype=np.float32, *args, **kwargs)


@typedispatch
def nan_like(grid: xu.UgridDataArray, *args, **kwargs):
    return xu.full_like(grid, fill_value=np.nan, dtype=np.float32, *args, **kwargs)


@typedispatch
def bounding_polygon(active: xr.DataArray):
    """Return bounding polygon of active cells"""
    # Force inactive cells to NaN.
    to_polygonize = active.where(active, other=np.nan)
    return polygonize(to_polygonize)


@typedispatch
def bounding_polygon(active: xu.UgridDataArray):
    """Return bounding polygon of active cells"""
    return active.ugrid.grid.bounding_polygon()


# Typedispatching doesn't work based on types of list elements, therefore to
# isinstance testing
def concat(grid_ls: list[GridDataArray | GridDataset], *args, **kwargs):
    if isinstance(grid_ls[0], (xu.UgridDataArray, xu.UgridDataset)):
        return xu.concat(grid_ls, *args, **kwargs)
    elif isinstance(grid_ls[0], (xr.DataArray, xr.Dataset)):
        return xr.concat(grid_ls, *args, **kwargs)
    else:
        raise TypeError(
            f"Expected type UgridDataArray or DataArray, received {type(grid_ls[0])}"
        )
