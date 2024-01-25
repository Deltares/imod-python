import textwrap
from typing import Callable, Sequence

import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.prepare import polygonize
from imod.typing import GridDataArray, GridDataset, structured


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
def nan_like(grid: xr.DataArray, dtype=np.float32, *args, **kwargs):
    return xr.full_like(grid, fill_value=np.nan, dtype=dtype, *args, **kwargs)


@typedispatch
def nan_like(grid: xu.UgridDataArray, dtype=np.float32, *args, **kwargs):
    return xu.full_like(grid, fill_value=np.nan, dtype=dtype, *args, **kwargs)


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
    first_object = _get_first_item(objects)
    start_type = type(first_object)
    homogeneous = all([isinstance(o, start_type) for o in objects])
    if not homogeneous:
        unique_types = set([type(o) for o in objects])
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


# Typedispatching doesn't work based on types of dict elements, therefore resort
# to manual type testing
def _type_dispatch_functions_on_dict(
    dict_of_objects: dict[str, GridDataArray | float | bool | int],
    unstructured_func: Callable,
    structured_func: Callable,
    *args,
    **kwargs,
):
    """
    Typedispatch function on grid and scalar variables provided in dictionary.
    Types do not need to be homogeneous as scalars and grids can be mixed. No
    mixing of structured and unstructured grids is allowed. Also allows running
    function on dictionary with purely scalars, in which case it will call to
    the xarray function.
    """

    error_msg = textwrap.dedent(
        """
        Received both xr.DataArray and xu.UgridDataArray. This means structured
        grids as well as unstructured grids were provided.
        """
    )

    if dict_of_objects is None:
        return xr.Dataset()

    types = [type(arg) for arg in dict_of_objects.values()]
    has_unstructured = xu.UgridDataArray in types
    has_structured = xr.DataArray in types
    if has_structured and has_unstructured:
        raise TypeError(error_msg)
    if has_unstructured:
        return unstructured_func([dict_of_objects], *args, **kwargs)

    return structured_func([dict_of_objects], *args, **kwargs)


def merge(
    objects: Sequence[GridDataArray | GridDataset], *args, **kwargs
) -> GridDataset:
    return _type_dispatch_functions_on_grid_sequence(
        objects, xu.merge, xr.merge, *args, **kwargs
    )


def merge_partitions(
    objects: Sequence[GridDataArray | GridDataset], *args, **kwargs
) -> GridDataArray | GridDataset:
    return _type_dispatch_functions_on_grid_sequence(
        objects, xu.merge_partitions, structured.merge_partitions, *args, **kwargs
    )


def concat(
    objects: Sequence[GridDataArray | GridDataset], *args, **kwargs
) -> GridDataArray | GridDataset:
    return _type_dispatch_functions_on_grid_sequence(
        objects, xu.concat, xr.concat, *args, **kwargs
    )


def merge_unstructured_dataset(variables_to_merge: list[dict], *args, **kwargs):
    """
    Work around xugrid issue https://github.com/Deltares/xugrid/issues/179

    Expects only one dictionary in list. List is used to have same API as
    xr.merge().

    Merges unstructured grids first, then manually assigns scalar variables.
    """
    if len(variables_to_merge) > 1:
        raise ValueError(
            f"Only one dict of variables expected, got {len(variables_to_merge)}"
        )

    variables_to_merge_dict = variables_to_merge[0]

    if not isinstance(variables_to_merge_dict, dict):
        raise TypeError(f"Expected dict, got {type(variables_to_merge_dict)}")

    # Separate variables into list of grids and dict of scalar variables
    grids_ls = []
    scalar_dict = {}
    for name, variable in variables_to_merge_dict.items():
        if isinstance(variable, xu.UgridDataArray):
            grids_ls.append(variable.rename(name))
        else:
            scalar_dict[name] = variable

    # Merge grids
    dataset = xu.merge(grids_ls, *args, **kwargs)

    # Assign scalar variables manually
    for name, variable in scalar_dict.items():
        dataset[name] = variable

    return dataset


def merge_with_dictionary(
    variables_to_merge: dict[str, GridDataArray | float | bool | int], *args, **kwargs
):
    return _type_dispatch_functions_on_dict(
        variables_to_merge, merge_unstructured_dataset, xr.merge, *args, **kwargs
    )


@typedispatch
def bounding_polygon(active: xr.DataArray):
    """Return bounding polygon of active cells"""
    to_polygonize = active.where(active, other=np.nan)
    polygons_gdf = polygonize(to_polygonize)
    # Filter polygons with inactive values (NaN)
    is_active_polygon = polygons_gdf["value"] == 1.0
    return polygons_gdf.loc[is_active_polygon]


@typedispatch
def bounding_polygon(active: xu.UgridDataArray):
    """Return bounding polygon of active cells"""
    active_indices = np.where(active > 0)[0]
    domain_slice = {f"{active.ugrid.grid.face_dimension}": active_indices}
    active_clipped = active.isel(domain_slice, missing_dims="ignore")

    return active_clipped.ugrid.grid.bounding_polygon()


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
