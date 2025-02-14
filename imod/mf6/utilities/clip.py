from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.mf6.interfaces.ilinedatapackage import ILineDataPackage
from imod.mf6.interfaces.ipackagebase import IPackageBase
from imod.mf6.interfaces.ipointdatapackage import IPointDataPackage
from imod.mf6.utilities.grid import get_active_domain_slice
from imod.prepare.hfb import clip_line_gdf_by_grid
from imod.typing import GeoDataFrameType, GridDataArray
from imod.typing.grid import is_spatial_grid
from imod.util.imports import MissingOptionalModule

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
