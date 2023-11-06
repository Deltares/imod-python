from __future__ import annotations

import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.mf6.interfaces.ilinedatapackage import ILineDataPackage
from imod.mf6.interfaces.ipackagebase import IPackageBase
from imod.mf6.interfaces.ipointdatapackage import IPointDataPackage
from imod.mf6.utilities.grid_utilities import get_active_domain_slice


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

    _filter_inactive_cells(clipped_package, active.sel(domain_slice))

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


@typedispatch
def clip_by_grid(
    _package: ILineDataPackage, _active: xr.DataArray | xu.UgridDataArray
) -> IPointDataPackage:
    """Clip LineDataPackage outside (un)structured grid."""
    raise NotImplementedError(
        "Clipping of line data packages ,e.g. hfb, is not supported"
    )


def _filter_inactive_cells(package, active):
    if package.is_grid_agnostic_package():
        return

    package_vars = package.dataset.data_vars
    for var in package_vars:
        if package_vars[var].shape != ():
            if np.issubdtype(package.dataset[var].dtype, np.integer):
                other = 0
            else:
                other = np.nan
            package.dataset[var] = package.dataset[var].where(active > 0, other=other)
