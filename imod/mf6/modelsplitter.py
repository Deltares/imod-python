from typing import Dict, List

import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.mf6.model import GroundwaterFlowModel, Modflow6Model
from imod.mf6.package import Package

DomainSlice = Dict[str, slice]


@typedispatch
def create_domain_slices(submodel_labels: xr.DataArray) -> List[DomainSlice]:
    _validate_submodel_label_array(submodel_labels)

    shape = submodel_labels.shape
    nrow, ncol = shape
    ds = xr.Dataset({"labels": submodel_labels})
    ds["column"] = (("y", "x"), np.broadcast_to(np.arange(ncol), shape))
    ds["row"] = (("y", "x"), np.broadcast_to(np.arange(nrow)[:, np.newaxis], shape))

    slices = []
    for _, group in ds.groupby("labels"):
        y_slice = slice(int(group["row"].min()), int(group["row"].max()) + 1)
        x_slice = slice(int(group["column"].min()), int(group["column"].max()) + 1)
        slices.append({"y": y_slice, "x": x_slice})

    return slices


@typedispatch
def create_domain_slices(
    submodel_labels: xu.UgridDataArray,
) -> List[DomainSlice]:
    _validate_submodel_label_array(submodel_labels)

    indices = xu.ugrid.partitioning.labels_to_indices(submodel_labels.values)
    slices = [{submodel_labels.ugrid.grid.face_dimension: index} for index in indices]
    return slices


def _validate_submodel_label_array(submodel_labels: xr.DataArray) -> None:
    unique_labels = np.unique(submodel_labels.values)

    if (
        len(unique_labels) == unique_labels.max() + 1
        and unique_labels.min() == 0
        and np.issubdtype(submodel_labels.dtype, np.integer)
    ):
        return
    raise ValueError(
        "The submodel_label  array should be integer and contain all the numbers between 0 and the number of "
        "partitions minus 1."
    )


def split_model(domain_slice: DomainSlice, model: Modflow6Model) -> Modflow6Model:
    """
    This function splits a Model into a number of submodels. The submodel_labels
    provided as input should have the same shape as a single layer of the model
    grid (all layers are split the same way), and contains an integer value in
    each cell. Each cell in the model grid will end up in the submodel with the
    index specified by the corresponding label of that cell. The labels should
    be numbers between 0 and the number of submodels.
    """
    new_model = GroundwaterFlowModel(**model._options)

    for pkg_name, package in model.items():
        new_model[pkg_name] = split_package(domain_slice, package)

    return new_model


def split_package(domain_slice: DomainSlice, package: Package):
    sliced_dataset = package.dataset.isel(domain_slice, missing_dims="ignore")
    return type(package)(**sliced_dataset)
