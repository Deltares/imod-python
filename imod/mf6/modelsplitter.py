from copy import deepcopy
from typing import Dict, List

import numpy as np
import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch

from imod.mf6.model import GroundwaterFlowModel, Modflow6Model

@typedispatch
def _partition(submodel_labels: xr.DataArray) -> List[Dict[str, slice]]:
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
def _partition(submodel_labels: xu.UgridDataArray) -> List[Dict[str, np.ndarray]]:
    indices = xu.ugrid.partitioning.labels_to_indices(submodel_labels.values)
    slices = [{submodel_labels.ugrid.grid.face_dimension: index} for index in indices]
    return slices

def _validate_label_array( label_array: xr.DataArray)-> None:
    unique_labels = np.unique(
        label_array.values
    )

    if  len(unique_labels) == unique_labels.max()+ 1 and unique_labels.min()==0 and np.issubdtype(label_array.dtype, np.integer):
        return
    raise ValueError("The label array should be integer and contain all the numbers between 0 and the number of partitions minus 1.")


def split_model_packages(  # noqa: F811
        submodel_labels: xu.UgridDataArray, model: Modflow6Model
) -> List[Modflow6Model]:

    """
    This function splits a Model into a number of submodels. The label_array
    provided as input should have the same shape as a single layer of the model
    grid (all layers are split the same way), and contains an integer value in
    each cell. Each cell in the model grid will end up in the submodel with the
    index specified by the corresponding label of that cell. The labels should
    be numbers between 0 and the number of submodels.
    """ 
    _validate_label_array(submodel_labels)  
    slices = _partition(submodel_labels)

    new_models = []
    for slice in slices:
        new_model = GroundwaterFlowModel(**model._options)

        for pkg_name, package in model.items():
            new_package = package.dataset.isel(slice, missing_dims="ignore")

            new_model[pkg_name] = new_package
        new_models.append(new_model)
    return new_models
