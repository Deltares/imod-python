from copy import deepcopy
from pathlib import Path
from typing import Tuple
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
from imod.mf6.pkgbase import Package
from imod.schemata import ValidationError
from imod.tests.fixtures.mf6_modelrun_fixture import assert_model_can_run


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


def test_masked_model_validation_inactive_cell_pillar(
    tmp_path: Path, unstructured_flow_model: GroundwaterFlowModel
):
    # create mask from idomain. Deactivate the same cell in all layers
    mask = unstructured_flow_model.get_domain()
    mask.loc[{"layer": 1, "mesh2d_nFaces": 23}] = 0
    mask.loc[{"layer": 2, "mesh2d_nFaces": 23}] = 0
    mask.loc[{"layer": 3, "mesh2d_nFaces": 23}] = 0
    unstructured_flow_model["disv"]["idomain"] = mask

    # apply the mask to a model
    unstructured_flow_model._mask_all_packages(mask)

    # test output validity
    errors = unstructured_flow_model._validate("model")
    assert len(errors.errors) == 0
    assert_model_can_run(unstructured_flow_model, "disv", tmp_path)


@pytest.mark.parametrize("layer_and_face", [(1, 23), (2, 23), (3, 23)])
def test_masked_model_validation_one_inactive_cell(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
    layer_and_face: Tuple[int, int],
):
    # create mask from idomain. a single cell
    layer, face = layer_and_face
    mask = unstructured_flow_model.get_domain()
    mask.loc[{"layer": layer, "mesh2d_nFaces": face}] = 0
    unstructured_flow_model["disv"]["idomain"] = mask

    # apply the mask to a model
    unstructured_flow_model._mask_all_packages(mask)

    # test output validity
    errors = unstructured_flow_model._validate("model")
    assert len(errors.errors) == 0
    assert_model_can_run(unstructured_flow_model, "disv", tmp_path)


@pytest.mark.parametrize("layer_and_face", [(1, 23), (2, 23), (3, 23)])
def test_masked_model_layered_and_scalar_package_input(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
    layer_and_face: Tuple[int, int],
):
    # Create mask from idomain. a single cell
    layer, face = layer_and_face
    mask = deepcopy(unstructured_flow_model.get_domain())
    mask.loc[{"layer": layer, "mesh2d_nFaces": face}] = 0

    # Make one package layer-based
    model_layers = np.array([1, 2, 3])
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": model_layers}, ("layer",))
    icelltype = xr.DataArray([1, 0, 0], {"layer": model_layers}, ("layer",))
    unstructured_flow_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=False,
        save_flows=True,
    )

    # Make one packages scalar-based
    unstructured_flow_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )

    # Apply the mask to a model
    unstructured_flow_model._mask_all_packages(mask)

    # Test output validity
    errors = unstructured_flow_model._validate("model")
    assert len(errors.errors) == 0
    assert_model_can_run(unstructured_flow_model, "disv", tmp_path)
