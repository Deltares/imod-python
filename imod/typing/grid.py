from typing import TypeAlias, Union

import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

GridDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]


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
