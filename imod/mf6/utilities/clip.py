from __future__ import annotations

from copy import deepcopy

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.mf6.interfaces.ilinedatapackage import ILineDataPackage
from imod.mf6.interfaces.ipackagebase import IPackageBase
from imod.mf6.interfaces.ipointdatapackage import IPointDataPackage
from imod.mf6.utilities.grid import get_active_domain_slice
from imod.mf6.utilities.hfb import (
    hfb_zlinestrings_to_zpolygons,
    hfb_zpolygons_to_zlinestrings,
)
from imod.typing import GridDataArray
from imod.typing.grid import bounding_polygon, is_spatial_grid
from imod.util.imports import MissingOptionalModule

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


@typedispatch  # type: ignore[no-redef]
def clip_by_grid(package: ILineDataPackage, active: GridDataArray) -> ILineDataPackage:  # noqa: F811
    """Clip LineDataPackage outside unstructured/structured grid."""
    clipped_line_data = clip_line_gdf_by_grid(package.line_data, active)

    # Create new instance
    clipped_package = deepcopy(package)
    clipped_package.line_data = clipped_line_data
    return clipped_package


def _clip_linestring(gdf_linestrings: gpd.GeoDataFrame, bounding_gdf: gpd.GeoDataFrame):
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


def clip_line_gdf_by_grid(
    gdf: gpd.GeoDataFrame, active: GridDataArray
) -> gpd.GeoDataFrame:
    """Clip GeoDataFrame by bounding polygon of grid"""
    # Clip line with polygon
    bounding_gdf = bounding_polygon(active)

    if gdf.has_z.any():
        # Shapely returns z linestrings when clipping our vertical z polygons.
        # To work around this convert polygons to zlinestrings to clip.
        # Consequently construct polygons from these clipped linestrings.
        gdf_linestrings = hfb_zpolygons_to_zlinestrings(gdf)
        clipped_linestrings = _clip_linestring(gdf_linestrings, bounding_gdf)
        return hfb_zlinestrings_to_zpolygons(clipped_linestrings)
    else:
        return _clip_linestring(gdf, bounding_gdf)
