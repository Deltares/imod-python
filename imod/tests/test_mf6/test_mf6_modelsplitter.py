import numpy as np
import pytest

from imod.mf6.modelsplitter import create_domain_slices, split_model
from imod.typing.grid import zeros_like


@pytest.mark.usefixtures("twri_model")
def test_partition_structured(twri_model):
    # create a partitioning array for the twri model.
    # Fill the first half of the array- in the y direction-  with 0's and the second half with 1's
    model = twri_model["GWF_1"]

    active = model["dis"]["idomain"].sel(layer=1)
    domain_center_y = model["dis"]["idomain"].coords["y"].mean()
    submodel_labels = zeros_like(active).where(active.coords["y"] > domain_center_y, 1)

    # split the model using the partitioning array
    model_slices = create_domain_slices(submodel_labels)
    new_models = [split_model(model_slice, model) for model_slice in model_slices]

    # verify result
    assert len(new_models) == 2

    # verify that the numbber of active cells in each submodel is the same as the number of labels
    unique_labels, label_counts = np.unique(submodel_labels.values, return_counts=True)
    for ilabel in unique_labels:
        imodel_idomain = (
            new_models[ilabel - 1]["dis"]["idomain"].sel({"layer": 1}).values
        )
        _, active_count = np.unique(imodel_idomain, return_counts=True)
        assert label_counts[ilabel - 1] == active_count[0]


@pytest.mark.usefixtures("circle_model")
def test_partition_unstructured(circle_model):
    model = circle_model["GWF_1"]

    # fill the first half of the array with 0's and the second half with 1
    active = model["disv"]["idomain"].sel(layer=1)
    x_bounds, y_bounds = np.array(active.grid.bounds).reshape(2, 2).T
    domain_center_y = y_bounds.mean()
    submodel_labels = zeros_like(active).where(active.grid.face_y > domain_center_y, 1)

    model_slices = create_domain_slices(submodel_labels)
    new_models = [split_model(model_slice, model) for model_slice in model_slices]

    # verify result
    assert len(new_models) == 2

    # verify that the numbber of active cells in each submodel is the same as the number of labels
    unique_labels, label_counts = np.unique(submodel_labels.values, return_counts=True)
    for ilabel in unique_labels:
        imodel_idomain = new_models[ilabel]["disv"]["idomain"].sel({"layer": 1}).values
        _, active_count = np.unique(imodel_idomain, return_counts=True)
        assert label_counts[ilabel] == active_count[0]
