import os
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import xugrid as xu

import imod
from imod.mf6.model import Modflow6Model
from imod.mf6.modelsplitter import PartitionInfo
from imod.mf6.statusinfo import StatusInfo
from imod.mf6.utilities.simulation_utilities import get_models, get_packages
from imod.schemata import ValidationError


def roundtrip(simulation, tmpdir_factory, name):
    # TODO: look at the values?
    tmp_path = tmpdir_factory.mktemp(name)
    simulation.dump(tmp_path)
    back = imod.mf6.Modflow6Simulation.from_file(tmp_path / f"{simulation.name}.toml")
    assert isinstance(back, imod.mf6.Modflow6Simulation)


@pytest.mark.usefixtures("twri_model")
def test_twri_roundtrip(twri_model, tmpdir_factory):
    roundtrip(twri_model, tmpdir_factory, "twri")


@pytest.mark.usefixtures("transient_twri_model")
def test_twri_transient_roundtrip(transient_twri_model, tmpdir_factory):
    roundtrip(transient_twri_model, tmpdir_factory, "twri_transient")


@pytest.mark.usefixtures("twri_disv_model")
def test_twri_disv_roundtrip(twri_disv_model, tmpdir_factory):
    roundtrip(twri_disv_model, tmpdir_factory, "twri_disv")


@pytest.mark.usefixtures("circle_model")
def test_circle_roundtrip(circle_model, tmpdir_factory):
    roundtrip(circle_model, tmpdir_factory, "circle")


@pytest.fixture(scope="function")
def setup_simulation():
    simulation = imod.mf6.Modflow6Simulation("TestSimulation")
    simulation["time_discretization"] = MagicMock(spec_set=imod.mf6.TimeDiscretization)

    solver_mock = MagicMock(spec=imod.mf6.Solution)
    solver_mock.__getitem__.side_effect = lambda x: getattr(solver_mock, x)
    solver_mock.modelnames = []

    simulation["solver"] = solver_mock

    return simulation


class TestModflow6Simulation:
    def test_write_with_default_arguments_writes_expected_files(
        self, tmp_path, setup_simulation
    ):
        # Arrange.
        simulation = setup_simulation

        # Act.
        simulation.write(tmp_path)

        # Assert.
        assert os.path.exists(os.path.join(tmp_path, "mfsim.nam"))

    def test_write_modflow6model_has_errors_raises_exception(
        self, tmp_path, setup_simulation
    ):
        # Arrange.
        simulation = setup_simulation

        model_status_info = StatusInfo()
        model_status_info.add_error("Test error")

        model_mock = MagicMock(spec_set=Modflow6Model)
        model_mock._model_id = "test_model_id"
        model_mock.write.return_value = model_status_info

        simulation["test_model"] = model_mock

        # Act/Assert.
        with pytest.raises(ValidationError):
            simulation.write(tmp_path)

    def test_split_simulation_only_has_packages(
        self, basic_unstructured_dis, setup_simulation
    ):
        # Arrange.
        idomain, top, bottom = basic_unstructured_dis

        simulation = setup_simulation
        simulation["disv"] = imod.mf6.VerticesDiscretization(
            top=top, bottom=bottom, idomain=idomain
        )
        simulation["solver"]["modelnames"] = []

        active = idomain.sel(layer=1)
        submodel_labels = xu.zeros_like(active).where(active.grid.face_y > 0.0, 1)

        # Act.
        new_simulation = simulation.split(submodel_labels)

        # Assert.
        assert len(get_models(new_simulation)) == 0
        assert len(get_packages(new_simulation)) == 3
        assert new_simulation["solver"] is simulation["solver"]
        assert (
            new_simulation["time_discretization"] is simulation["time_discretization"]
        )
        assert new_simulation["disv"] is simulation["disv"]

    @mock.patch("imod.mf6.simulation.slice_model", autospec=True)
    def test_split_multiple_models(
        self, slice_model_mock, basic_unstructured_dis, setup_simulation
    ):
        # Arrange.
        idomain, top, bottom = basic_unstructured_dis

        simulation = setup_simulation

        model_mock1 = MagicMock(spec_set=Modflow6Model)
        model_mock1._model_id = "test_model_id1"

        model_mock2 = MagicMock(spec_set=Modflow6Model)
        model_mock2._model_id = "test_model_id2"

        simulation["test_model1"] = model_mock1
        simulation["test_model2"] = model_mock2

        simulation["solver"]["modelnames"] = ["test_model1", "test_model2"]

        slice_model_mock.return_value = MagicMock(spec_set=Modflow6Model)

        active = idomain.sel(layer=1)
        submodel_labels = xu.zeros_like(active).where(active.grid.face_y > 0.0, 1)

        # Act.
        new_simulation = simulation.split(submodel_labels)

        # Assert.
        new_models = get_models(new_simulation)
        assert slice_model_mock.call_count == 4
        assert len(new_models) == 4

        # fmt: off
        assert len([model_name for model_name in new_models.keys() if "test_model1" in model_name]) == 2
        assert len([model_name for model_name in new_models.keys() if "test_model2" in model_name]) == 2

        active_domain1 = submodel_labels.where(submodel_labels == 0, -1).where(submodel_labels != 0, 1)
        active_domain2 = submodel_labels.where(submodel_labels == 1, -1).where(submodel_labels != 1, 1)
        # fmt: on

        expected_slice_model_calls = [
            (PartitionInfo(id=0, active_domain=active_domain1), model_mock1),
            (PartitionInfo(id=0, active_domain=active_domain1), model_mock2),
            (PartitionInfo(id=1, active_domain=active_domain2), model_mock1),
            (PartitionInfo(id=1, active_domain=active_domain2), model_mock2),
        ]

        for expected_call in expected_slice_model_calls:
            assert any(
                compare_submodel_partition_info(expected_call[0], call_args[0][0])
                and (expected_call[1] is call_args[0][1])
                for call_args in slice_model_mock.call_args_list
            )


def compare_submodel_partition_info(first: PartitionInfo, second: PartitionInfo):
    return (first.id == second.id) and np.array_equal(
        first.active_domain, second.active_domain
    )
