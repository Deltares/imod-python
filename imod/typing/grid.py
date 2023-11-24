from typing import Callable, Sequence

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
def is_unstructured(grid: xu.UgridDataArray | xu.UgridDataset) -> bool:
    return True


@typedispatch
def is_unstructured(grid: xr.DataArray | xr.Dataset) -> bool:
    return False


def _force_decreasing_y(structured_grid: xr.DataArray | xr.Dataset):
    flip = slice(None, None, -1)
    if structured_grid.indexes["y"].is_monotonic_increasing:
        structured_grid = structured_grid.isel(y=flip)
    elif not structured_grid.indexes["y"].is_monotonic_decreasing:
        raise RuntimeError(
            f"Non-monotonous y-coordinates for grid: {structured_grid.name}."
        )
    return structured_grid


def _force_object_sequence(objects: Sequence | dict):
    if isinstance(objects[0], dict):
        return objects[0].values()
    else:
        return objects


def _get_first_item(objects: Sequence):
    return next(iter(objects))


# Typedispatching doesn't work based on types of list elements, therefore resort to
# isinstance testing
def _type_dispatch_functions_on_grid_sequence(
    objects: Sequence[GridDataArray | GridDataset],
    unstructured_func: Callable,
    structured_func: Callable,
    *args,
    **kwargs,
) -> GridDataArray | GridDataset:
    """
    Type dispatch functions on sequence of grids. Functions like merging or concatenating.
    """
    object_sequence = _force_object_sequence(objects)
    first_object = _get_first_item(object_sequence)
    start_type = type(first_object)
    homogeneous = all([isinstance(o, start_type) for o in object_sequence])
    if not homogeneous:
        unique_types = set([type(o) for o in object_sequence])
        raise TypeError(
            f"Only homogeneous sequences can be reduced, received sequence of {unique_types}"
        )
    if isinstance(first_object, (xu.UgridDataArray, xu.UgridDataset)):
        return unstructured_func(objects, *args, **kwargs)
    elif isinstance(first_object, (xr.DataArray, xr.Dataset)):
        return _force_decreasing_y(structured_func(objects, *args, **kwargs))
    raise TypeError(
        f"'{unstructured_func.__name__}' not supported for type {type(objects[0])}"
    )


def merge(
    objects: Sequence[GridDataArray | GridDataset | dict], *args, **kwargs
) -> GridDataset:
    return _type_dispatch_functions_on_grid_sequence(
        objects, xu.merge, xr.merge, *args, **kwargs
    )


def merge_partitions(
    objects: Sequence[GridDataArray | GridDataset], *args, **kwargs
) -> GridDataArray | GridDataset:
    return _type_dispatch_functions_on_grid_sequence(
        objects, xu.merge_partitions, xr.merge, *args, **kwargs
    )


def concat(
    objects: Sequence[GridDataArray | GridDataset], *args, **kwargs
) -> GridDataArray | GridDataset:
    return _type_dispatch_functions_on_grid_sequence(
        objects, xu.concat, xr.concat, *args, **kwargs
    )


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
