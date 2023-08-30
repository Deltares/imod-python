from copy import deepcopy
from typing import Dict, List

import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.mf6.model import GroundwaterFlowModel, Modflow6Model


def _partition_structured_slices(labels: xr.DataArray) -> List[Dict[str, slice]]:
    """
    This function creates "slices" (tuples of start index, stop index and step
    size as a built-in python class for array slicing) that describe areas in a
    2d array that have the same label. the labels are given as input in a 2d
    data grid.
    """
    shape = labels.shape
    nrow, ncol = shape
    ds = xr.Dataset({"labels": labels})
    ds["column"] = (("y", "x"), np.broadcast_to(np.arange(ncol), shape))
    ds["row"] = (("y", "x"), np.broadcast_to(np.arange(nrow)[:, np.newaxis], shape))

    slices = []
    for _, group in ds.groupby("labels"):
        y_slice = slice(int(group["row"].min()), int(group["row"].max()) + 1)
        x_slice = slice(int(group["column"].min()), int(group["column"].max()) + 1)
        slices.append({"y": y_slice, "x": x_slice})

    return slices

def _validate_label_array( label_array: xr.DataArray)-> None:
    
    unique_labels, label_counts = np.unique(
        label_array.values, return_counts=True
    )

    if  len(unique_labels) == unique_labels.max()+ 1 and unique_labels.min()==0 and np.issubdtype(label_array.dtype, np.integer):
        return
    raise ValueError("The label array should be integer and contain all the numbers between 0 and the number of partitions minus 1.")


@typedispatch
def split_model_packages(
    submodel_labels: xr.DataArray, model: Modflow6Model
) -> List[Modflow6Model]:
    """
    This function splits a structured Model into a number of submodels. The
    label_array provided as input should have the same shape as a single layer
    of the model grid (all layers are split in an identical fashion) and
    contains an integer value in each cell. Each cell in the model grid will end
    up in the submodel with the index specified by the corresponding label of
    that cell. The labels should be numbers between 0 and the number of
    submodels.
    """
    _validate_label_array(submodel_labels)
    slices = _partition_structured_slices(submodel_labels)
    new_models = []
    for slice in slices:
        new_model = GroundwaterFlowModel(**model._options)
        for pkg_name, package in model.items():
            new_package_dataset = package.dataset.isel(slice, missing_dims="ignore")
            package_type = type(package)
            new_model[pkg_name] = package_type (**new_package_dataset)

        new_models.append(new_model)
    return new_models


@typedispatch
def split_model_packages(  # noqa: F811
    submodel_labels: xu.UgridDataArray, model: Modflow6Model
) -> List[Modflow6Model]:
    """
    This function splits an unstructured Model into a number of submodels. The
    label_array provided as input should have the same shape as a single layer
    of the model grid (all layers are split in an identical fashion) and
    contains an integer value in each cell. Each cell in the model grid will end
    up in the submodel with the index specified by the corresponding label of
    that cell. The labels should be numbers between 0 and the number of
    submodels.
    """
    _validate_label_array(submodel_labels)
    indices = xu.ugrid.partitioning.labels_to_indices(submodel_labels.values)
    indexes = [(submodel_labels.ugrid.grid.face_dimension, index) for index in indices]

    new_models = []
    for dimname, index in indexes:
        new_model = GroundwaterFlowModel(**model._options)

        for pkg_name, package in model.items():
            new_package = package.dataset.isel({dimname: index}, missing_dims="ignore")

            new_model[pkg_name] = new_package
        new_models.append(new_model)
    return new_models
