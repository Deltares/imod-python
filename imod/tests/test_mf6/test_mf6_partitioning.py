import copy

import numpy as np
import pytest

import imod
from imod.mf6.modelsplitter import split_model_packages


@pytest.mark.usefixtures("twri_model")
def test_partition_structured(twri_model):
    # create a partitioning array for the twri model.
    # Fill the first half of the array- in the y direction-  with 0's and the second half with 1
    partitioning_array = copy.deepcopy(
        twri_model["GWF_1"]["dis"]["idomain"].sel({"layer": 1})
    )
    number_of_subdomains = 2

    ny, nx = partitioning_array.shape

    for isub in range(number_of_subdomains):
        from_index = round(ny / number_of_subdomains) * (isub)
        to_index = round(ny / number_of_subdomains) * (isub + 1)
        for ix in range(nx):
            for iy in range(ny):
                if iy >= from_index and iy < to_index:
                    partitioning_array.values[iy, ix] = isub

    # split the model using the partitioning array
    new_models = split_model_packages(partitioning_array, twri_model["GWF_1"])

    # verify result
    assert len(new_models) == number_of_subdomains

    # verify that the numbber of active cells in each submodel is the same as the number of labels
    unique_labels, label_counts = np.unique(
        partitioning_array.values, return_counts=True
    )
    for ilabel in unique_labels:
        imodel_idomain = (
            new_models[ilabel - 1]["dis"]["idomain"].sel({"layer": 1}).values
        )
        _, active_count = np.unique(imodel_idomain, return_counts=True)
        assert label_counts[ilabel - 1] == active_count[0]


@pytest.mark.usefixtures("circle_model")
def test_partition_unstructured(circle_model, tmp_path):
    partitioning_array = copy.deepcopy(
        circle_model["GWF_1"]["disv"]["idomain"].sel({"layer": 1})
    )

    # fill the first half of the array with 0's and the second half with 1

    ncell = partitioning_array.shape[0]
    for icell in range(ncell):
        if icell < round(ncell / 2):
            label = 0
        else:
            label = 1
        partitioning_array.values[icell] = label

    new_models = split_model_packages(partitioning_array, circle_model["GWF_1"])

    # verify result
    assert len(new_models) == 2

    # verify that the numbber of active cells in each submodel is the same as the number of labels
    unique_labels, label_counts = np.unique(
        partitioning_array.values, return_counts=True
    )
    for ilabel in unique_labels:
        imodel_idomain = new_models[ilabel]["disv"]["idomain"].sel({"layer": 1}).values
        _, active_count = np.unique(imodel_idomain, return_counts=True)
        assert label_counts[ilabel] == active_count[0]
