from __future__ import annotations

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.mf6.interfaces.ilinedatapackage import ILineDataPackage
from imod.mf6.interfaces.ipackagebase import IPackageBase
from imod.mf6.interfaces.ipointdatapackage import IPointDataPackage
from imod.mf6.utilities.dataset_utilities import get_scalar_variables
from imod.mf6.utilities.grid_utilities import get_active_domain_slice
from imod.prepare import polygonize


@typedispatch
def clip_by_grid(_: object, grid: object) -> None:
    raise TypeError(
        f"'grid' should be of type xr.DataArray, xu.Ugrid2d or xu.UgridDataArray, got {type(grid)}"
    )


@typedispatch
def clip_by_grid(package: IPackageBase, active: xr.DataArray) -> IPackageBase:
    domain_slice = get_active_domain_slice(active)
    x_min, x_max = domain_slice["x"].start, domain_slice["x"].stop
    y_min, y_max = domain_slice["y"].stop, domain_slice["y"].start

    clipped_package = package.clip_box(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )

    if "idomain" in package.dataset:
        clipped_package.dataset["idomain"] = xr.ones_like(
            clipped_package.dataset["idomain"]
        ) * active.sel(domain_slice)

    return clipped_package


@typedispatch
def clip_by_grid(package: IPackageBase, active: xu.UgridDataArray) -> IPackageBase:
    domain_slice = get_active_domain_slice(active)

    clipped_dataset = package.dataset.isel(domain_slice, missing_dims="ignore")
    if "idomain" in package.dataset:
        idomain = package.dataset["idomain"]
        clipped_dataset["idomain"] = idomain.sel(clipped_dataset["idomain"].indexes)

    cls = type(package)
    new = cls.__new__(cls)
    new.dataset = clipped_dataset
    return new


@typedispatch
def clip_by_grid(
    package: IPointDataPackage, active: xu.UgridDataArray
) -> IPointDataPackage:
    """Clip PointDataPackage outside unstructured grid."""
    points = np.column_stack((package.x, package.y))

    is_inside_exterior = active.grid.locate_points(points) != -1
    selection = package.dataset.loc[{"index": is_inside_exterior}]

    cls = type(package)
    new = cls.__new__(cls)
    new.dataset = selection
    return new


def __get_settings(package):
    scalar_variables = get_scalar_variables(package.dataset)
    return package[scalar_variables]


def __get_variables_for_gdf(package: ILineDataPackage):
    return [
        package._get_variable_name(),
        "geometry",
    ] + package._get_vertical_variables()


def __line_package_to_gdf(package: ILineDataPackage):
    variables_for_gdf = __get_variables_for_gdf(package)
    return gpd.GeoDataFrame(
        package.dataset[variables_for_gdf].to_dataframe(),
        geometry="geometry",
    )


@typedispatch
def clip_by_grid(
    package: ILineDataPackage, active: xu.UgridDataArray
) -> ILineDataPackage:
    """Clip LineDataPackage outside unstructured grid."""

    # Convert package to Geopandas' GeoDataFrame
    package_gdf = __line_package_to_gdf(package)
    # Clip line with polygon
    bounding_polygon = active.ugrid.grid.bounding_polygon()
    package_gdf_clipped = package_gdf.clip(bounding_polygon)
    # Get settings
    settings = __get_settings(package)
    # Create new instance
    cls = type(package)
    return cls(package_gdf_clipped, **settings)


@typedispatch
def clip_by_grid(package: ILineDataPackage, active: xr.DataArray) -> ILineDataPackage:
    """Clip LineDataPackage outside structured grid."""

    # Convert package to Geopandas' GeoDataFrame
    package_gdf = __line_package_to_gdf(package)
    # Clip line with polygon
    bounding_polygon = polygonize(active.where(active, other=np.nan))
    package_gdf_clipped = package_gdf.clip(bounding_polygon)
    # Get settings
    settings = __get_settings(package)
    # Create new instance
    cls = type(package)
    return cls(package_gdf_clipped, **settings)
