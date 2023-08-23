import xarray as xr
from typing import List, Dict
import numpy as np
from imod.mf6.model import Modflow6Model, GroundwaterFlowModel
from copy import deepcopy
import xugrid as xu

def partition_structured_slices(labels: xr.DataArray) -> List[Dict[str, slice]]:
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


def split_model_packages( label_array, model: Modflow6Model ):
    slices = partition_structured_slices(label_array)
    new_models = []
    for slice in slices:
        new_model = GroundwaterFlowModel(**model._options)
        for pack_name, package in model.items():
           
            new_package_dataset = package.dataset.isel(slice, missing_dims = "ignore")
            new_package = deepcopy(package)
            new_package.dataset = new_package_dataset

            new_model[pack_name] = new_package
        new_models.append( new_model)
    return new_models



def split_model_unstructured_packages( label_array, model: Modflow6Model ):

    indices = xu.ugrid.partitioning.labels_to_indices(label_array.values)
    indexes = [(label_array.ugrid.grid.face_dimension, index) for index in indices]

    new_models = []
    for dimname, index in indexes:
        new_model = GroundwaterFlowModel(**model._options)

        for pack_name, package in model.items():
           
            new_package = package.dataset.isel({dimname: index}, missing_dims = "ignore")

            new_model[pack_name] = new_package
        new_models.append( new_model)
    return new_models

    '''new_models = []
    nr_new_models = label_array.values.max() + 1
    for imodel in range (nr_new_models+1):
        new_models.append(GroundwaterFlowModel(**model._options))

    for pack_name, package in model.items():




        if "mesh2d_nFaces" in model[pack_name].dataset.coords :
            new_packages = package.dataset.ugrid.partition_by_label(label_array)
            for imodel in range (nr_new_models):
                new_models[imodel][pack_name] =  model[pack_name].__class__(**new_packages[imodel])            
        else:
            for imodel in range (nr_new_models):
                new_models[imodel][pack_name] =  deepcopy( package)      
    return new_models
    '''








    

