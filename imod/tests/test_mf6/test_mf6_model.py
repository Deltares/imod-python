from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from jinja2 import Template
from xugrid.core.wrap import UgridDataArray

import imod
from imod.mf6 import ConstantHead
from imod.mf6.model import GroundwaterFlowModel, Modflow6Model
from imod.mf6.package import Package
from imod.schemata import ValidationError


# Duplicate from test_mf6_dis.py
# Probably move to fixtures
@pytest.fixture(scope="function")
def idomain_and_bottom():
    nlay = 3
    nrow = 15
    ncol = 15
    shape = (nlay, nrow, ncol)

    dx = 5000.0
    dy = -5000.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}
    idomain = xr.DataArray(np.ones(shape, dtype=np.int8), coords=coords, dims=dims)
    bottom = xr.DataArray([-200.0, -350.0, -450.0], {"layer": layer}, ("layer",))

    return idomain, bottom


def test_checks_required_pkgs(idomain_and_bottom):
    idomain, bottom = idomain_and_bottom

    gwf_model = imod.mf6.GroundwaterFlowModel()

    # Case 1: All packages present
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=200.0, bottom=bottom, idomain=idomain
    )
    gwf_model["ic"] = imod.mf6.InitialConditions(start=0.0)
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(0, 10.0)
    gwf_model["sto"] = imod.mf6.SpecificStorage(1e-5, 0.1, True, 0)
    gwf_model["oc"] = imod.mf6.OutputControl()

    gwf_model._check_for_required_packages("GWF_1")

    # Case 2: Output Control package missing
    gwf_model.pop("oc")

    with pytest.raises(ValueError, match="No oc package found in model GWF_1"):
        gwf_model._check_for_required_packages("GWF_1")

    # Case 3: DIS package missing
    gwf_model["oc"] = imod.mf6.OutputControl()
    gwf_model.pop("dis")

    with pytest.raises(
        ValueError, match="No dis/disv/disu package found in model GWF_1"
    ):
        gwf_model._check_for_required_packages("GWF_1")


def test_key_assign():
    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["ic"] = imod.mf6.InitialConditions(start=0.0)

    with pytest.raises(KeyError):
        gwf_model["way-too-long-key-name"] = imod.mf6.InitialConditions(start=0.0)


def roundtrip(model, tmp_path):
    model.dump(tmp_path, "test")
    back = type(model).from_file(tmp_path / "test/test.toml")
    assert isinstance(back, type(model))


@pytest.mark.usefixtures("circle_model")
def test_circle_roundtrip(circle_model, tmp_path):
    roundtrip(circle_model["GWF_1"], tmp_path)


class TestModel:
    def test_write_valid_model_without_error(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"

        sut = Modflow6Model()

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"

        sut["dis"] = discretization_mock

        template_mock = MagicMock(spec_set=Template)
        template_mock.render.return_value = ""
        sut._template = template_mock

        global_times_mock = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        status = sut.write(tmp_path, model_name, global_times_mock)

        # Assert.
        assert not status.has_errors()

    def test_write_without_dis_pkg_return_error(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"

        sut = Modflow6Model()

        template_mock = MagicMock(spec_set=Template)
        template_mock.render.return_value = ""
        sut._template = template_mock

        global_times_mock = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        status = sut.write(tmp_path, model_name, global_times_mock)

        # Assert.
        assert status.has_errors()

    def test_write_with_invalid_pkg_returns_error(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"

        sut = Modflow6Model()

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"
        discretization_mock._validate.return_value = {
            "test_var": [ValidationError("error_string")]
        }

        sut["dis"] = discretization_mock

        template_mock = MagicMock(spec_set=Template)
        template_mock.render.return_value = ""
        sut._template = template_mock

        global_times_mock = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        status = sut.write(tmp_path, model_name, global_times_mock)

        # Assert.
        assert status.has_errors()

    def test_write_with_two_invalid_pkg_returns_two_errors(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"

        sut = Modflow6Model()

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"
        discretization_mock._validate.return_value = {
            "test1_var": [ValidationError("error_string1")]
        }

        package_mock = MagicMock(spec_set=Package)
        package_mock._pkg_id = "test_package"
        package_mock._validate.return_value = {
            "test2_var": [ValidationError("error_string2")]
        }

        sut["dis"] = discretization_mock
        sut["test_package"] = package_mock

        template_mock = MagicMock(spec_set=Template)
        template_mock.render.return_value = ""
        sut._template = template_mock

        global_times_mock = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        status = sut.write(tmp_path, model_name, global_times_mock)

        # Assert.
        assert len(status.errors) == 2


class TestGroundwaterFlowModel:
    def test_clip_box_without_state_for_boundary(self):
        # Arrange.
        state_for_boundary = None

        sut = GroundwaterFlowModel()

        # Act.
        clipped = sut.clip_box(state_for_boundary=state_for_boundary)

        # Assert.
        assert "chd_clipped" not in clipped

    @mock.patch("imod.mf6.model.ClippedBoundaryConditionCreator")
    def test_clip_box_with_state_for_boundary(self, clipped_boundary_condition_creator):
        # Arrange.
        state_for_boundary = MagicMock(spec_set=UgridDataArray)

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"
        discretization_mock.clip_box.return_value = discretization_mock

        clipped_boundary_condition_creator.create.side_effect = [
            None,
            MagicMock(spec_set=ConstantHead),
        ]
        clipped_boundary_condition_creator.return_value = (
            clipped_boundary_condition_creator
        )

        sut = GroundwaterFlowModel()
        sut["dis"] = discretization_mock

        # Act.
        clipped = sut.clip_box(state_for_boundary=state_for_boundary)

        # Assert.
        assert "chd_clipped" in clipped
        clipped_boundary_condition_creator.create.assert_called_with(
            discretization_mock["idomain"],
            state_for_boundary,
            [],
        )

    @mock.patch("imod.mf6.model.ClippedBoundaryConditionCreator")
    def test_clip_box_with_unassigned_boundaries_in_original_model(
        self, clipped_boundary_condition_creator
    ):
        # Arrange.
        state_for_boundary = MagicMock(spec_set=UgridDataArray)

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"
        discretization_mock.clip_box.return_value = discretization_mock

        constant_head_mock = MagicMock(spec_set=ConstantHead)
        constant_head_mock.clip_box.return_value = constant_head_mock

        unassigned_boundary_constant_head_mock = MagicMock(spec_set=ConstantHead)

        clipped_boundary_condition_creator.create.side_effect = [
            unassigned_boundary_constant_head_mock,
            MagicMock(spec_set=ConstantHead),
        ]
        clipped_boundary_condition_creator.return_value = (
            clipped_boundary_condition_creator
        )

        sut = GroundwaterFlowModel()
        sut["dis"] = discretization_mock
        sut["chd"] = constant_head_mock

        # Act.
        clipped = sut.clip_box(state_for_boundary=state_for_boundary)

        # Assert.
        assert "chd_clipped" in clipped
        clipped_boundary_condition_creator.create.assert_called_with(
            discretization_mock["idomain"],
            state_for_boundary,
            [constant_head_mock, unassigned_boundary_constant_head_mock],
        )
