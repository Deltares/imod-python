import pickle
import textwrap
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    ParamSpec,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np
import xarray as xr
import xugrid as xu
from plum import Dispatcher

from imod.typing import (
    GeoDataFrameType,
    GridDataArray,
    GridDataset,
    structured,
)
from imod.util.imports import MissingOptionalModule
from imod.util.spatial import _polygonize

# create dispatcher instance to limit scope of typedispatching
dispatch = Dispatcher()

T = TypeVar("T")
P = ParamSpec("P")

if TYPE_CHECKING:
    import geopandas as gpd
else:
    try:
        import geopandas as gpd
    except ImportError:
        gpd = MissingOptionalModule("geopandas")


@dispatch
def zeros_like(grid: xr.DataArray, *args, **kwargs):
    return xr.zeros_like(grid, *args, **kwargs)


@dispatch  # type: ignore[no-redef]
def zeros_like(grid: xu.UgridDataArray, *args, **kwargs):  # noqa: F811
    return xu.zeros_like(grid, *args, **kwargs)


@dispatch
def ones_like(grid: xr.DataArray, *args, **kwargs):
    return xr.ones_like(grid, *args, **kwargs)


@dispatch  # type: ignore[no-redef]
def ones_like(grid: xu.UgridDataArray, *args, **kwargs):  # noqa: F811
    return xu.ones_like(grid, *args, **kwargs)


@dispatch
def nan_like(grid: xr.DataArray, dtype=np.float32, *args, **kwargs):
    return xr.full_like(grid, fill_value=np.nan, dtype=dtype, *args, **kwargs)


@dispatch  # type: ignore[no-redef]
def nan_like(grid: xu.UgridDataArray, dtype=np.float32, *args, **kwargs):  # noqa: F811
    return xu.full_like(grid, fill_value=np.nan, dtype=dtype, *args, **kwargs)


@dispatch
def full_like(grid: xr.DataArray, fill_value, *args, **kwargs):
    return xr.full_like(grid, fill_value, *args, **kwargs)


@dispatch  # type: ignore [no-redef]
def full_like(grid: xu.UgridDataArray, fill_value, *args, **kwargs):  # noqa: F811
    return xu.full_like(grid, fill_value, *args, **kwargs)


@dispatch
def is_unstructured(grid: xu.UgridDataArray | xu.UgridDataset) -> bool:
    return True


@dispatch  # type: ignore[no-redef]
def is_unstructured(grid: xr.DataArray | xr.Dataset) -> bool:  # noqa: F811
    return False


@dispatch  # type: ignore[no-redef]
def is_unstructured(grid: Any) -> bool:  # noqa: F811
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
    homogeneous = all(isinstance(o, start_type) for o in objects)
    if not homogeneous:
        unique_types = {type(o) for o in objects}
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
    dict_of_objects: Mapping[str, GridDataArray | float | bool | int],
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
        Received both structured grid (xr.DataArray) and xu.UgridDataArray. This
        means structured grids as well as unstructured grids were provided.
        """
    )

    if dict_of_objects is None:
        return xr.Dataset()

    types = [type(arg) for arg in dict_of_objects.values()]
    has_unstructured = xu.UgridDataArray in types
    # Test structured if xr.DataArray and spatial.
    has_structured_grid = any(
        isinstance(arg, xr.DataArray) and is_spatial_grid(arg)
        for arg in dict_of_objects.values()
    )
    if has_structured_grid and has_unstructured:
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

    # Temporarily work around this xugrid issue, until fixed:
    # https://github.com/Deltares/xugrid/issues/206
    grid_hashes = [hash(pickle.dumps(grid)) for grid in dataset.ugrid.grids]
    unique_grid_hashes = np.unique(grid_hashes)
    if unique_grid_hashes.size > 1:
        raise ValueError(
            "Multiple grids provided, please provide data on one unique grid"
        )
    else:
        # Possibly won't work anymore if this ever gets implemented:
        # https://github.com/Deltares/xugrid/issues/195
        dataset._grids = [dataset.grids[0]]

    # Assign scalar variables manually
    for name, variable in scalar_dict.items():
        dataset[name] = variable

    return dataset


def merge_with_dictionary(
    variables_to_merge: Mapping[str, GridDataArray | float | bool | int],
    *args,
    **kwargs,
):
    return _type_dispatch_functions_on_dict(
        variables_to_merge, merge_unstructured_dataset, xr.merge, *args, **kwargs
    )


@dispatch
def bounding_polygon(active: xr.DataArray) -> GeoDataFrameType:
    """Return bounding polygon of active cells"""
    to_polygonize = active.where(active, other=np.nan)
    polygons_gdf = _polygonize(to_polygonize)
    # Filter polygons with inactive values (NaN)
    is_active_polygon = polygons_gdf["value"] > 0
    return polygons_gdf.loc[is_active_polygon]


@dispatch  # type: ignore[no-redef]
def bounding_polygon(active: xu.UgridDataArray) -> GeoDataFrameType:  # noqa: F811
    """Return bounding polygon of active cells"""
    active_indices = np.where(active > 0)[0]
    domain_slice = {f"{active.ugrid.grid.face_dimension}": active_indices}
    active_clipped = active.isel(domain_slice, missing_dims="ignore")
    polygon = active_clipped.ugrid.grid.bounding_polygon()
    dummy_value = 0
    return gpd.GeoDataFrame([dummy_value], geometry=[polygon])


@dispatch
def is_spatial_grid(array: xr.DataArray | xr.Dataset) -> bool:
    """Return True if the array contains data in at least 2 spatial dimensions"""
    coords = array.coords
    dims = array.dims
    has_spatial_coords = "x" in coords and "y" in coords
    has_spatial_dims = "x" in dims and "y" in dims
    return has_spatial_coords & has_spatial_dims


@dispatch  # type: ignore[no-redef]
def is_spatial_grid(array: xu.UgridDataArray | xu.UgridDataset) -> bool:  # noqa: F811
    """Return True if the array contains data associated to cell faces"""
    face_dim = array.ugrid.grid.face_dimension
    dims = array.dims
    coords = array.coords
    has_spatial_coords = face_dim in coords
    has_spatial_dims = face_dim in dims
    return has_spatial_dims & has_spatial_coords


@dispatch  # type: ignore[no-redef]
def is_spatial_grid(_: Any) -> bool:  # noqa: F811
    return False


@dispatch
def is_equal(array1: xu.UgridDataArray, array2: xu.UgridDataArray) -> bool:
    return array1.equals(array2) and array1.ugrid.grid.equals(array2.ugrid.grid)


@dispatch  # type: ignore[no-redef]
def is_equal(array1: xr.DataArray, array2: xr.DataArray) -> bool:  # noqa: F811
    return array1.equals(array2)


@dispatch  # type: ignore[no-redef]
def is_equal(array1: Any, array2: Any) -> bool:  # noqa: F811
    return False


@dispatch
def is_same_domain(grid1: xu.UgridDataArray, grid2: xu.UgridDataArray) -> bool:
    return grid1.coords.equals(grid2.coords) and grid1.ugrid.grid.equals(
        grid2.ugrid.grid
    )


@dispatch  # type: ignore[no-redef]
def is_same_domain(grid1: xr.DataArray, grid2: xr.DataArray) -> bool:  # noqa: F811
    return grid1.coords.equals(grid2.coords)


@dispatch  # type: ignore[no-redef]
def is_same_domain(grid1: Any, grid2: Any) -> bool:  # noqa: F811
    return False


@dispatch
def get_spatial_dimension_names(grid: xr.DataArray) -> list[str]:
    return ["x", "y", "layer", "dx", "dy"]


@dispatch  # type: ignore[no-redef]
def get_spatial_dimension_names(grid: xu.UgridDataArray) -> list[str]:  # noqa: F811
    facedim = grid.ugrid.grid.face_dimension
    return [facedim, "layer"]


@dispatch  # type: ignore[no-redef]
def get_spatial_dimension_names(grid: Any) -> list[str]:  # noqa: F811
    return []


def get_non_spatial_dimension_names(grid: GridDataArray) -> list[str]:
    spatial_dims = get_spatial_dimension_names(grid)
    return [str(dim) for dim in grid.dims if dim not in spatial_dims]


@dispatch
def get_grid_geometry_hash(grid: xr.DataArray) -> tuple[int, int]:
    hash_x = hash(pickle.dumps(grid["x"].values))
    hash_y = hash(pickle.dumps(grid["y"].values))
    return (hash_x, hash_y)


@dispatch  # type: ignore[no-redef]
def get_grid_geometry_hash(grid: xu.UgridDataArray) -> tuple[int, int, Any]:  # noqa: F811
    hash_x = hash(pickle.dumps(grid.ugrid.grid.node_x))
    hash_y = hash(pickle.dumps(grid.ugrid.grid.node_y))
    hash_connectivity = hash(pickle.dumps(grid.ugrid.grid.node_face_connectivity))
    return (hash_x, hash_y, hash_connectivity)


@dispatch  # type: ignore[no-redef]
def get_grid_geometry_hash(grid: Any) -> tuple[int, int]:  # noqa: F811
    raise ValueError("get_grid_geometry_hash not supported for this object.")


@dispatch
def enforce_dim_order(grid: xr.DataArray) -> xr.DataArray:
    """Enforce dimension order to iMOD Python standard"""
    return grid.transpose("species", "time", "layer", "y", "x", missing_dims="ignore")


@dispatch  # type: ignore[no-redef]
def enforce_dim_order(grid: xu.UgridDataArray) -> xu.UgridDataArray:  # noqa: F811
    """Enforce dimension order to iMOD Python standard"""
    face_dimension = grid.ugrid.grid.face_dimension
    return grid.transpose(
        "species", "time", "layer", face_dimension, missing_dims="ignore"
    )


@dispatch  # type: ignore[no-redef]
def enforce_dim_order(grid: None) -> None:  # noqa: F811
    return grid


@dispatch  # type: ignore[no-redef]
def enforce_dim_order(grid: Any) -> xr.DataArray:  # noqa: F811
    """Enforce dimension order to iMOD Python standard"""
    raise TypeError(f"Function doesn't support type {type(grid)}")


def _as_ugrid_dataarray_with_topology(
    obj: GridDataArray, topology: xu.Ugrid2d
) -> xu.UgridDataArray:
    """Force obj and topology to ugrid dataarray"""
    return xu.UgridDataArray(xr.DataArray(obj), topology)


def preserve_gridtype(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to preserve gridtype, this is to work around the following xugrid
    behavior:

    >>> UgridDataArray() * DataArray() -> UgridDataArray
    >>> DataArray() * UgridDataArray() -> DataArray

    with this decorator:

    >>> UgridDataArray() * DataArray() -> UgridDataArray
    >>> DataArray() * UgridDataArray() -> UgridDataArray
    """

    @wraps(func)
    def decorator(*args: P.args, **kwargs: P.kwargs):
        unstructured = False
        grid = None
        for arg in args:
            if is_unstructured(arg):
                unstructured = True
                ds = cast(xu.UgridDataArray | xu.UgridDataset, arg)
                grid = ds.ugrid.grid

        x = func(*args, **kwargs)

        if unstructured:
            # Multiple grids returned
            if isinstance(x, tuple):
                return tuple(_as_ugrid_dataarray_with_topology(i, grid) for i in x)
            return _as_ugrid_dataarray_with_topology(x, grid)
        return x

    return decorator


@dispatch
def is_empty(obj: xr.Dataset) -> bool:
    return len(obj.keys()) == 0


@dispatch  # type: ignore[no-redef]
def is_empty(obj: Any) -> bool:  # noqa: F811
    return False


def is_planar_grid(
    grid: xr.DataArray | xr.Dataset | xu.UgridDataArray | xu.UgridDataset,
) -> bool:
    # Returns True if the grid is planar.
    # A grid is considered planar when:
    # - it is a spatial grid (x, y coordinates or cellface/edge coordinates)
    # - it has no layer coordinates, or, it has a single layer coordinate with
    #   value less or equal to zero
    if not is_spatial_grid(grid):
        return False
    if "layer" not in grid.coords:
        return True
    if grid["layer"].shape == ():
        return True
    if grid["layer"][0] <= 0 and len(grid["layer"]) == 1:
        return True
    return False


def has_negative_layer(
    grid: xr.DataArray | xr.Dataset | xu.UgridDataArray | xu.UgridDataset,
) -> bool:
    if not is_spatial_grid(grid):
        return False
    if "layer" not in grid.coords:
        return False
    if grid["layer"].shape == ():
        return False
    if grid["layer"][0] < 0:
        return True
    return False


def is_transient_data_grid(
    grid: xr.DataArray | xr.Dataset | xu.UgridDataArray | xu.UgridDataset,
):
    # Returns True if there is a time coordinate on the object with more than one value.
    if "time" in grid.coords:
        if len(grid["time"]) > 1:
            return True
    return False


class GridCache:
    """
    Cache grids in this object for a specific function, lookup grids based on
    unique geometry hash.
    """

    def __init__(self, func: Callable, max_cache_size=5):
        self.max_cache_size = max_cache_size
        self.grid_cache: dict[int, GridDataArray] = {}
        self.func = func

    def get_grid(self, grid: GridDataArray):
        geom_hash = get_grid_geometry_hash(grid)
        if geom_hash not in self.grid_cache.keys():
            if len(self.grid_cache.keys()) >= self.max_cache_size:
                self.remove_first()
            self.grid_cache[geom_hash] = self.func(grid)
        return self.grid_cache[geom_hash]

    def remove_first(self):
        keys = list(self.grid_cache.keys())
        self.grid_cache.pop(keys[0])

    def clear(self):
        self.grid_cache = {}


UGRID2D_FROM_STRUCTURED_CACHE = GridCache(xu.Ugrid2d.from_structured)


@dispatch
def as_ugrid_dataarray(grid: xr.DataArray) -> xu.UgridDataArray:
    """
    Enforce GridDataArray to UgridDataArray, calls
    xu.UgridDataArray.from_structured, which is a costly operation. Therefore
    cache results.
    """

    topology = UGRID2D_FROM_STRUCTURED_CACHE.get_grid(grid)

    # Copied from:
    # https://github.com/Deltares/xugrid/blob/3dee693763da1c4c0859a4f53ac38d4b99613a33/xugrid/core/wrap.py#L236
    # Note that "da" is renamed to "grid" and "grid" to "topology"
    dims = grid.dims[:-2]
    coords = {k: grid.coords[k] for k in dims}
    face_da = xr.DataArray(
        grid.data.reshape(*grid.shape[:-2], -1),
        coords=coords,
        dims=[*dims, topology.face_dimension],
        name=grid.name,
    )
    return xu.UgridDataArray(face_da, topology)


@dispatch  # type: ignore[no-redef]
def as_ugrid_dataarray(grid: xu.UgridDataArray) -> xu.UgridDataArray:  # noqa: F811
    """Enforce GridDataArray to UgridDataArray"""
    return grid


@dispatch  # type: ignore[no-redef]
def as_ugrid_dataarray(grid: Any) -> xu.UgridDataArray:  # noqa: F811
    raise TypeError(f"Function doesn't support type {type(grid)}")


def notnull(obj: GridDataArray) -> GridDataArray:
    """
    Helper function; does the same as xr.DataArray.notnull. This function is to
    avoid an issue where xr.DataArray.notnull() returns ordinary numpy arrays
    for instances of xu.UgridDataArray.
    """

    return cast(GridDataArray, ~np.isnan(obj))
