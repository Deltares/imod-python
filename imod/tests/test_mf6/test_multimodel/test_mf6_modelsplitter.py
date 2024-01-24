import numpy as np
import pytest

from imod.mf6.multimodel.modelsplitter import create_partition_info, slice_model
from imod.tests.fixtures.mf6_modelrun_fixture import assert_simulation_can_run
from imod.typing.grid import zeros_like


@pytest.mark.usefixtures("twri_model")
def test_slice_model_structured(twri_model):
    """
    Create a sub model array for the twri model.
    Fill the first half of the array, in the y-direction, with 0's and the second half with 1's
    """

    # Arrange.
    model = twri_model["GWF_1"]

    active = model.domain.sel(layer=1)
    domain_center_x = active.coords["x"].mean()
    domain_center_y = active.coords["y"].mean()
    submodel_labels = zeros_like(active).where(
        (active.coords["x"] > domain_center_x) & (active.coords["y"] > domain_center_y),
        1,
    )

    # Act.
    partition_info = create_partition_info(submodel_labels)
    new_models = [
        slice_model(model_partition_info, model)
        for model_partition_info in partition_info
    ]

    # Assert.
    assert len(new_models) == 2

    # verify that the number of active cells in each submodel is the same as the number of labels
    unique_labels, label_counts = np.unique(submodel_labels.values, return_counts=True)
    for submodel_label in unique_labels:
        active = new_models[submodel_label].domain.sel({"layer": 1})
        active_count = active.where(active > 0).count()
        assert label_counts[submodel_label] == active_count.values


@pytest.mark.usefixtures("circle_model")
def test_slice_model_unstructured(circle_model):
    """
    Create a sub model array for the circle model.
    Fill the first half of the array, in the y-direction, with 0's and the second half with 1's
    """

    # Arrange.
    model = circle_model["GWF_1"]

    active = model.domain.sel(layer=1)
    _, y_bounds = np.array(active.grid.bounds).reshape(2, 2).T
    domain_center_y = y_bounds.mean()
    submodel_labels = zeros_like(active).where(active.grid.face_y > domain_center_y, 1)

    # Act.
    partition_info = create_partition_info(submodel_labels)
    new_models = [
        slice_model(model_partition_info, model)
        for model_partition_info in partition_info
    ]

    # Assert.
    assert len(new_models) == 2

    # verify that the number of active cells in each submodel is the same as the number of labels
    unique_labels, label_counts = np.unique(submodel_labels.values, return_counts=True)
    for submodel_label in unique_labels:
        active_count = new_models[submodel_label].domain.sel({"layer": 1}).count()
        assert label_counts[submodel_label] == active_count.values


@pytest.mark.usefixtures("flow_transport_simulation")
def test_slice_model_with_auxiliary_variables(tmp_path, flow_transport_simulation):
    flow_simulation = flow_transport_simulation
    flow_simulation.pop("tpt_a")
    flow_simulation.pop("tpt_b")
    flow_simulation.pop("tpt_c")
    flow_simulation.pop("tpt_d")
    flow_simulation.pop("transport_solver")

    flow_model = flow_simulation["flow"]
    active = flow_model.domain

    submodel_labels = zeros_like(active)
    submodel_labels = submodel_labels.drop_vars("layer")
    submodel_labels.values[:, :, 50:] = 1
    submodel_labels = submodel_labels.sel(layer=0, drop=True)

    flow_simulation.write(tmp_path, binary=False)
    split_simulation = flow_simulation.split(submodel_labels)
    split_simulation.write(tmp_path, binary=False)
    assert_simulation_can_run(split_simulation, "dis", tmp_path)
    pass
