from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.mf6.model import GroundwaterFlowModel, Modflow6Model
from imod.mf6.package import Package
from imod.typing.grid import GridDataArray

DomainSlice = Dict[str, slice]


@dataclass
class SubmodelPartitionInfo:
    active_domain: GridDataArray
    label_id: int = -1
    slice: DomainSlice = None


@typedispatch
def create_partition_info(submodel_labels: xr.DataArray) -> List[SubmodelPartitionInfo]:
    """
    A DomainSlice is used to partition a model or package. The domain slices are created using a submodel_labels
    array. The submodel_labels provided as input should have the same shape as a single layer of the model grid (all
    layers are split the same way), and contains an integer value in each cell. Each cell in the model grid will end
    up in the submodel with the index specified by the corresponding label of that cell. The labels should be numbers
    between 0 and the number of submodels.
    """
    _validate_submodel_label_array(submodel_labels)

    shape = submodel_labels.shape
    nrow, ncol = shape
    ds = xr.Dataset({"labels": submodel_labels})
    ds["column"] = (("y", "x"), np.broadcast_to(np.arange(ncol), shape))
    ds["row"] = (("y", "x"), np.broadcast_to(np.arange(nrow)[:, np.newaxis], shape))

    submodel_partition_infos = []
    for label_id, group in ds.groupby("labels"):
        y_slice = slice(int(group["row"].min()), int(group["row"].max()) + 1)
        x_slice = slice(int(group["column"].min()), int(group["column"].max()) + 1)
        partition_slice = {"y": y_slice, "x": x_slice}

        active_domain = submodel_labels.where(submodel_labels.values == label_id).isel(
            partition_slice
        )
        active_domain = xr.where(active_domain.notnull(),1,-1).astype(submodel_labels.dtype)

        submodel_partition_info = SubmodelPartitionInfo(
            label_id=label_id, slice=partition_slice, active_domain=active_domain
        )

        submodel_partition_infos.append(submodel_partition_info)

    return submodel_partition_infos


@typedispatch
def create_partition_info(
    submodel_labels: xu.UgridDataArray,
) -> List[SubmodelPartitionInfo]:
    """
    A DomainSlice is used to partition a model or package. The domain slices are created using a submodel_labels
    array. The submodel_labels provided as input should have the same shape as a single layer of the model grid (all
    layers are split the same way), and contains an integer value in each cell. Each cell in the model grid will end
    up in the submodel with the index specified by the corresponding label of that cell. The labels should be numbers
    between 0 and the number of submodels.
    """
    _validate_submodel_label_array(submodel_labels)

    unique_labels, label_counts = np.unique(submodel_labels.values, return_counts=True)
    indices = xu.ugrid.partitioning.labels_to_indices(submodel_labels.values)

    submodel_partition_infos = []
    for label_id, index in zip(unique_labels, indices):
        partition_slice = {submodel_labels.ugrid.grid.face_dimension: index}
        active_domain = submodel_labels.where(submodel_labels.values == label_id).isel(
            partition_slice
        )
        active_domain = xr.where(active_domain.notnull(), 1, -1).astype(submodel_labels.dtype)

        submodel_partition_info = SubmodelPartitionInfo(
            label_id=label_id, slice=partition_slice, active_domain=active_domain
        )

        submodel_partition_infos.append(submodel_partition_info)

    return submodel_partition_infos


def _validate_submodel_label_array(submodel_labels: GridDataArray) -> None:
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


def slice_model(
    partition_info: SubmodelPartitionInfo, model: Modflow6Model
) -> Modflow6Model:
    """
    This function slices a Modflow6Model.  A sliced model is a model that consists of packages of the original model
    that are sliced using the domain_slice. A domain_slice can be created using the
    :func:`imod.mf6.modelsplitter.create_domain_slices` function.
    """
    new_model = GroundwaterFlowModel(**model._options)

    for pkg_name, package in model.items():
        new_model[pkg_name] = _slice_package(partition_info, package)

    return new_model


def _slice_package(partition_info: SubmodelPartitionInfo, package: Package) -> Package:
    sliced_dataset = package.dataset.isel(partition_info.slice, missing_dims="ignore")

    if "idomain" in package.dataset:
        idomain = package.dataset["idomain"]
        idomain_type = idomain.dtype
        sliced_dataset["idomain"] = (
            idomain.isel(partition_info.slice, missing_dims="ignore")
            .where(partition_info.active_domain > 0, -1)
            .astype(idomain_type)
        )

    return type(package)(**sliced_dataset)
