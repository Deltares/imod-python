import os
from unittest.mock import MagicMock

import pytest

import imod
from imod.mf6.model import Modflow6Model
from imod.mf6.statusinfo import StatusInfo
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


class TestModflow6Simulation:
    def test_write_with_default_arguments_writes_expected_files(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")

        sut = imod.mf6.Modflow6Simulation("TestSimulation")
        sut["time_discretization"] = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        sut.write(tmp_path)

        # Assert.
        assert os.path.exists(os.path.join(tmp_path, "mfsim.nam"))

    def test_write_modflow6model_has_errors_raises_exception(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")

        sut = imod.mf6.Modflow6Simulation("TestSimulation")
        sut["time_discretization"] = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        model_status_info = StatusInfo()
        model_status_info.add_error("Test error")

        model_mock = MagicMock(spec_set=Modflow6Model)
        model_mock._model_id = "test_model_id"
        model_mock.write.return_value = model_status_info

        sut["test_model"] = model_mock

        # Act/Assert.
        with pytest.raises(ValidationError):
            sut.write(tmp_path)
