from __future__ import annotations

from copy import deepcopy

import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.mf6.interfaces.ilinedatapackage import ILineDataPackage
from imod.mf6.interfaces.ipackagebase import IPackageBase
from imod.mf6.interfaces.ipointdatapackage import IPointDataPackage
from imod.mf6.utilities.grid import get_active_domain_slice
from imod.typing import GridDataArray
from imod.typing.grid import bounding_polygon, is_spatial_2D
from imod.util.imports import MissingOptionalModule

try:
    import shapely
except ImportError:
    shapely = MissingOptionalModule("shapely")


@typedispatch  # type: ignore [no-redef]
def clip_by_grid(_: object, grid: object) -> None:
    raise TypeError(
        f"'grid' should be of type xr.DataArray, xu.Ugrid2d or xu.UgridDataArray, got {type(grid)}"
    )


@typedispatch  # type: ignore [no-redef]
def clip_by_grid(package: IPackageBase, active: xr.DataArray) -> IPackageBase:  # noqa: F811
    domain_slice = get_active_domain_slice(active)
    x_min, x_max = domain_slice["x"].start, domain_slice["x"].stop
    y_min, y_max = domain_slice["y"].stop, domain_slice["y"].start

    clipped_package = package.clip_box(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )

    _filter_inactive_cells(clipped_package, active.sel(domain_slice))

    return clipped_package


@typedispatch  # type: ignore [no-redef]
def clip_by_grid(package: IPackageBase, active: xu.UgridDataArray) -> IPackageBase:  # noqa: F811
    domain_slice = get_active_domain_slice(active)

    clipped_dataset = package.dataset.isel(domain_slice, missing_dims="ignore")

    cls = type(package)
    new = cls.__new__(cls)
    new.dataset = clipped_dataset
    return new


@typedispatch  # type: ignore [no-redef]
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
    new = cls.__new__(cls)
    new.dataset = selection
    return new


def _filter_inactive_cells(package, active):
    if package.is_grid_agnostic_package():
        return

    package_vars = package.dataset.data_vars
    for var in package_vars:
        if package_vars[var].shape != ():
            if is_spatial_2D(package.dataset[var]):
                if np.issubdtype(package.dataset[var].dtype, np.integer):
                    other = 0
                else:
                    other = np.nan
                package.dataset[var] = package.dataset[var].where(
                    active > 0, other=other
                )


@typedispatch  # type: ignore [no-redef]
def clip_by_grid(package: ILineDataPackage, active: GridDataArray) -> ILineDataPackage:  # noqa: F811
    """Clip LineDataPackage outside unstructured/structured grid."""

    # Clip line with polygon
    bounding_gdf = bounding_polygon(active)
    clipped_line_data = package.line_data.clip(bounding_gdf)

    # Catch edge case: when line crosses only vertex of polygon, a point
    # or multipoint is returned. Drop these.
    type_ids = shapely.get_type_id(clipped_line_data.geometry)
    is_points = (type_ids == shapely.GeometryType.POINT) | (
        type_ids == shapely.GeometryType.MULTIPOINT
    )
    clipped_line_data = clipped_line_data[~is_points]

    # Convert MultiLineStrings to LineStrings
    clipped_line_data = clipped_line_data.explode("geometry", ignore_index=True)

    # Create new instance
    clipped_package = deepcopy(package)
    clipped_package.line_data = clipped_line_data
    return clipped_package
