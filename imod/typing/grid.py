from typing import List, TypeAlias, Union

import numpy as np
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


@typedispatch
def nan_like(grid: xr.DataArray, *args, **kwargs):
    return xr.full_like(grid, fill_value=np.nan, dtype=np.float32, *args, **kwargs)


@typedispatch
def nan_like(grid: xu.UgridDataArray, *args, **kwargs):
    return xu.full_like(grid, fill_value=np.nan, dtype=np.float32, *args, **kwargs)


@typedispatch
def is_unstructured(grid: xu.UgridDataArray) -> bool:
    return True


@typedispatch
def is_unstructured(grid: xr.DataArray) -> bool:
    return False


@typedispatch
def merge(*args: xr.DataArray) -> xr.DataArray:
    return xr.merge(list(args))


@typedispatch
def merge(*args: xu.UgridDataArray) -> xu.UgridDataArray:
    xu.Ugrid2d.merge_partitions(list(args))
    return xu.merge(list(args),  compat = "override")
