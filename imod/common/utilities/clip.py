from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.common.interfaces.ilinedatapackage import ILineDataPackage
from imod.common.interfaces.ipackagebase import IPackageBase
from imod.common.interfaces.ipointdatapackage import IPointDataPackage
from imod.common.utilities.grid import get_active_domain_slice
from imod.common.utilities.line_data import (
    clipped_zbound_linestrings_to_vertical_polygons,
    vertical_polygons_to_zbound_linestrings,
)
from imod.common.utilities.value_filters import is_valid
from imod.typing import GeoDataFrameType, GridDataArray, GridDataset
from imod.typing.grid import bounding_polygon, is_spatial_grid
from imod.util.imports import MissingOptionalModule
from imod.util.time import to_datetime_internal

if TYPE_CHECKING:
    import geopandas as gpd
else:
    try:
        import geopandas as gpd
    except ImportError:
        gpd = MissingOptionalModule("geopandas")

try:
    import shapely
except ImportError:
    shapely = MissingOptionalModule("shapely")


@typedispatch
def clip_by_grid(_: object, grid: object) -> None:
    raise TypeError(
        f"'grid' should be of type xr.DataArray, xu.Ugrid2d or xu.UgridDataArray, got {type(grid)}"
    )


@typedispatch  # type: ignore[no-redef]
def clip_by_grid(package: IPackageBase, active: xr.DataArray) -> IPackageBase:  # noqa: F811
    domain_slice = get_active_domain_slice(active)
    x_min, x_max = domain_slice["x"].start, domain_slice["x"].stop
    y_min, y_max = domain_slice["y"].stop, domain_slice["y"].start

    clipped_package = package.clip_box(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )

    _filter_inactive_cells(clipped_package, active.sel(domain_slice))

    return clipped_package


@typedispatch  # type: ignore[no-redef]
def clip_by_grid(package: IPackageBase, active: xu.UgridDataArray) -> IPackageBase:  # noqa: F811
    domain_slice = get_active_domain_slice(active)

    clipped_dataset = package.dataset.isel(domain_slice, missing_dims="ignore")

    cls = type(package)
    return cls._from_dataset(clipped_dataset)


@typedispatch  # type: ignore[no-redef]
def clip_by_grid(  # noqa: F811
    package: IPointDataPackage, active: xu.UgridDataArray
) -> IPointDataPackage:
    """Clip PointDataPackage outside unstructured grid."""

    domain_slice = get_active_domain_slice(active)
    active_clipped = active.isel(domain_slice, missing_dims="ignore")

    points = np.column_stack((package.x, package.y))

    is_inside_exterior = active_clipped.grid.locate_points(points) != -1
    selection = package.dataset.loc[{"index": is_inside_exterior}]

    cls = type(package)
    return cls._from_dataset(selection)


def _filter_inactive_cells(package, active):
    if package.is_grid_agnostic_package():
        return

    package_vars = package.dataset.data_vars
    for var in package_vars:
        if package_vars[var].shape != ():
            if is_spatial_grid(package.dataset[var]):
                other = (
                    0
                    if np.issubdtype(package.dataset[var].dtype, np.integer)
                    else np.nan
                )

                package.dataset[var] = package.dataset[var].where(
                    active > 0, other=other
                )


@typedispatch  # type: ignore[no-redef, misc]
def clip_by_grid(package: ILineDataPackage, active: GridDataArray) -> ILineDataPackage:  # noqa: F811
    """Clip LineDataPackage outside unstructured/structured grid."""
    clipped_line_data = clip_line_gdf_by_grid(package.line_data, active)

    # Create new instance
    clipped_package = deepcopy(package)
    clipped_package.line_data = clipped_line_data
    return clipped_package


def _clip_linestring(
    gdf_linestrings: GeoDataFrameType, bounding_gdf: GeoDataFrameType
) -> GeoDataFrameType:
    clipped_line_data = gdf_linestrings.clip(bounding_gdf)

    # Catch edge case: when line crosses only vertex of polygon, a point
    # or multipoint is returned. Drop these.
    type_ids = shapely.get_type_id(clipped_line_data.geometry)
    is_points = (type_ids == shapely.GeometryType.POINT) | (
        type_ids == shapely.GeometryType.MULTIPOINT
    )
    clipped_line_data = clipped_line_data[~is_points]

    if clipped_line_data.index.shape[0] == 0:
        # Shortcut if GeoDataFrame is empty
        return clipped_line_data

    # Convert MultiLineStrings to LineStrings, index parts of MultiLineStrings
    clipped_line_data = clipped_line_data.explode(
        "geometry", ignore_index=False, index_parts=True
    )
    if clipped_line_data.index.nlevels == 3:
        index_names = ["bound", "index", "parts"]
    else:
        index_names = ["index", "parts"]
    clipped_line_data.index = clipped_line_data.index.set_names(index_names)
    return clipped_line_data


def clip_line_gdf_by_bounding_polygon(
    gdf: GeoDataFrameType, bounding_gdf: GeoDataFrameType
) -> GeoDataFrameType:
    if (shapely.get_type_id(gdf.geometry) == shapely.GeometryType.POLYGON).any():
        # Shapely returns z linestrings when clipping our vertical z polygons.
        # To work around this convert polygons to zlinestrings to clip.
        # Consequently construct polygons from these clipped linestrings.
        gdf_linestrings = vertical_polygons_to_zbound_linestrings(gdf)
        clipped_linestrings = _clip_linestring(gdf_linestrings, bounding_gdf)
        return clipped_zbound_linestrings_to_vertical_polygons(clipped_linestrings)
    else:
        return _clip_linestring(gdf, bounding_gdf)


def clip_line_gdf_by_grid(
    gdf: GeoDataFrameType, active: GridDataArray
) -> GeoDataFrameType:
    """Clip GeoDataFrame by bounding polygon of grid"""
    # Clip line with polygon
    bounding_gdf = bounding_polygon(active)
    return clip_line_gdf_by_bounding_polygon(gdf, bounding_gdf)


def bounding_polygon_from_line_data_and_clip_box(
    line_data: GeoDataFrameType,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> GeoDataFrameType:
    line_minx, line_miny, line_maxx, line_maxy = line_data.total_bounds
    # Use pandas clip, as it gracefully deals with lower=None and upper=None
    x_min, x_max = pd.Series([line_minx, line_maxx]).clip(lower=x_min, upper=x_max)
    y_min, y_max = pd.Series([line_miny, line_maxy]).clip(lower=y_min, upper=y_max)
    bbox = shapely.box(x_min, y_min, x_max, y_max)
    dummy_value = 0
    return gpd.GeoDataFrame([dummy_value], geometry=[bbox])


def clip_time_indexer(
    time: np.ndarray,
    time_start: Optional[cftime.datetime | np.datetime64 | str] = None,
    time_end: Optional[cftime.datetime | np.datetime64 | str] = None,
):
    """
    Return indices which can be used to select a time slice from a
    DataArray or Dataset.
    """
    original = xr.DataArray(
        data=np.arange(time.size),
        coords={"time": time},
        dims=("time",),
    )
    indexer = original.sel(time=slice(time_start, time_end))

    # The selection might return a 0-sized dimension.
    if indexer.size > 0:
        first_time = indexer["time"].values[0]
    else:
        first_time = None

    # If the first time matches exactly, xarray will have done thing we
    # wanted and our work with the time dimension is finished.
    if (time_start is None) or (time_start == first_time):
        return indexer

    # If the first time is before the original time, we need to
    # backfill; otherwise, we need to ffill the first timestamp.
    if time_start < time[0]:
        method = "bfill"
    else:
        method = "ffill"
    # Index with a list rather than a scalar to preserve the time
    # dimension.
    first = original.sel(time=[time_start], method=method)
    first["time"] = [time_start]
    indexer = xr.concat([first, indexer], dim="time")

    return indexer


def clip_spatial_box(
    dataset: GridDataset,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    """
    Clip a spatial dataset by a bounding box.
    """
    selection = dataset
    x_slice = slice(x_min, x_max)
    y_slice = slice(y_min, y_max)
    if isinstance(selection, xu.UgridDataset):
        selection = selection.ugrid.sel(x=x_slice, y=y_slice)
    elif ("x" in selection.coords) and ("y" in selection.coords):
        if selection.indexes["y"].is_monotonic_decreasing:
            y_slice = slice(y_max, y_min)
        selection = selection.sel(x=x_slice, y=y_slice)
    return selection


def clip_layer_slice(
    dataset: GridDataset,
    layer_min: Optional[int] = None,
    layer_max: Optional[int] = None,
):
    """
    Clip a dataset by a layer slice.
    """
    selection = dataset
    if "layer" in selection.coords:
        layer_slice = slice(layer_min, layer_max)
        # Cannot select if it's not a dimension!
        if "layer" not in selection.dims:
            selection = (
                selection.expand_dims("layer").sel(layer=layer_slice).squeeze("layer")
            )
        else:
            selection = selection.sel(layer=layer_slice)
    return selection


def _to_datetime(
    time: Optional[cftime.datetime | np.datetime64 | str], use_cftime: bool
):
    """
    Helper function that converts to datetime, except when None.
    """
    if time is None:
        return time
    else:
        return to_datetime_internal(time, use_cftime)


def _is_within_timeslice(
    keys: np.ndarray,
    time_start: Optional[cftime.datetime | np.datetime64 | str] = None,
    time_end: Optional[cftime.datetime | np.datetime64 | str] = None,
) -> np.ndarray:
    """
    Return a boolean array indicating whether the keys are within the time slice.
    """
    within_time_slice = np.ones(keys.size, dtype=bool)
    if time_start is not None:
        within_time_slice &= keys >= time_start
    if time_end is not None:
        within_time_slice &= keys <= time_end
    return within_time_slice


def clip_repeat_stress(
    repeat_stress: xr.DataArray,
    time: np.ndarray,
    time_start: Optional[cftime.datetime | np.datetime64 | str] = None,
    time_end: Optional[cftime.datetime | np.datetime64 | str] = None,
):
    """
    Selection may remove the original data which are repeated.
    These should be re-inserted at the first occuring "key".
    Next, remove these keys as they've been "promoted" to regular
    timestamps with data.

    Say repeat_stress is:
        keys       : values
        2001-01-01 : 2000-01-01
        2002-01-01 : 2000-01-01
        2003-01-01 : 2000-01-01

    And time_start = 2001-01-01, time_end = None

    This function returns:
        keys       : values
        2002-01-01 : 2001-01-01
        2003-01-01 : 2001-01-01
    """
    # First, "pop" and filter.
    keys, values = repeat_stress.values.T
    # Prepend unique values to keys to account for "value" entries
    # that need to be clipped off.
    unique_values = np.unique(values)
    prepended_keys = np.concatenate((unique_values, keys))
    prepended_values = np.concatenate((unique_values, values))
    # Clip timeslice
    within_time_slice = _is_within_timeslice(prepended_keys, time_start, time_end)
    clipped_keys = prepended_keys[within_time_slice]
    clipped_values = prepended_values[within_time_slice]
    # Now account for "value" entries that have been clipped off, these should
    # be updated in the end to ``insert_keys``.
    insert_values, index = np.unique(clipped_values, return_index=True)
    insert_keys = clipped_keys[index]
    # Setup indexer
    indexer = xr.DataArray(
        data=np.arange(time.size),
        coords={"time": time},
        dims=("time",),
    ).sel(time=insert_values)
    indexer["time"] = insert_keys

    # Update the key-value pairs. Discard keys that have been "promoted" to
    # values.
    not_promoted = np.isin(clipped_keys, insert_keys, assume_unique=True, invert=True)
    not_promoted_keys = clipped_keys[not_promoted]
    not_promoted_values = clipped_values[not_promoted]
    # Promote the values to their new source.
    to_promote = np.searchsorted(insert_values, not_promoted_values)
    promoted_values = insert_keys[to_promote]
    repeat_stress = xr.DataArray(
        data=np.column_stack((not_promoted_keys, promoted_values)),
        dims=("repeat", "repeat_items"),
    )
    return indexer, repeat_stress


def clip_time_slice(
    dataset: GridDataset,
    time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
    time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
):
    """Clip time slice from dataset, account for repeat stress if present."""
    selection = dataset
    if "time" in selection.coords:
        time = selection["time"].values
        use_cftime = isinstance(time[0], cftime.datetime)
        time_start = _to_datetime(time_min, use_cftime)
        time_end = _to_datetime(time_max, use_cftime)

        indexer = clip_time_indexer(
            time=time,
            time_start=time_start,
            time_end=time_end,
        )

        if "repeat_stress" in selection.data_vars and is_valid(
            selection["repeat_stress"].values[()]
        ):
            repeat_indexer, repeat_stress = clip_repeat_stress(
                repeat_stress=selection["repeat_stress"],
                time=time,
                time_start=time_start,
                time_end=time_end,
            )
            selection = selection.drop_vars("repeat_stress")
            selection["repeat_stress"] = repeat_stress
            indexer = repeat_indexer.combine_first(indexer).astype(int)

        selection = selection.drop_vars("time").isel(time=indexer)
    return selection
