from typing import Sequence, TypeAlias, Union

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


def merge(objects: Sequence[xr.DataArray], *args, **kwargs) -> xr.Dataset:
    start_type = type(objects[0])
    homogeneous = all([isinstance(o, start_type) for o in objects])
    if not homogeneous:
        raise RuntimeError("only hommogeneous sequences can be merged")
    if isinstance(objects[0], xr.DataArray):
        return xr.merge(objects, *args, **kwargs)
    if isinstance(objects[0], xu.UgridDataArray):
        return xu.merge_partitions(objects, *args, **kwargs)
    raise NotImplementedError(f"merging not supported for type {type(objects[0])}")


def merge_to_dataset(objects: Sequence[xr.DataArray], *args, **kwargs) -> xr.Dataset:
    start_type = type(objects[0])
    homogeneous = all([isinstance(o, start_type) for o in objects])
    if not homogeneous:
        raise RuntimeError("only hommogeneous sequences can be merged")
    if isinstance(objects[0], xr.DataArray):
        return xr.merge(objects, *args, **kwargs)
    if isinstance(objects[0], xu.UgridDataArray):
        return xu.merge(objects, *args, **kwargs)
    raise NotImplementedError(f"merging not supported for type {type(objects[0])}")
