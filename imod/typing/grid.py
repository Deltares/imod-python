from typing import Sequence

import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.prepare import polygonize


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


@typedispatch
def is_spatial_2D(array: xr.DataArray) -> bool:
    """Return True if the array contains data in at least 2 spatial dimensions"""
    coords = array.coords
    dims = array.dims
    has_spatial_coords = "x" in coords and "y" in coords
    has_spatial_dims = "x" in dims and "y" in dims
    return has_spatial_coords & has_spatial_dims


@typedispatch
def is_spatial_2D(array: xu.UgridDataArray) -> bool:
    """Return True if the array contains data associated to cell faces"""
    face_dim = array.ugrid.grid.face_dimension
    dims = array.dims
    coords = array.coords
    has_spatial_coords = face_dim in coords
    has_spatial_dims = face_dim in dims
    return has_spatial_dims & has_spatial_coords
