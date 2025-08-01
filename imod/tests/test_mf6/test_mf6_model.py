from copy import deepcopy
from unittest import mock
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr
from jinja2 import Template
from xugrid.core.wrap import UgridDataArray

import imod
from imod.mf6 import ConstantHead
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.mf6.model import Modflow6Model
from imod.mf6.model_gwf import GroundwaterFlowModel
from imod.mf6.package import Package
from imod.mf6.validation_settings import ValidationSettings
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.typing.grid import concat, nan_like


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


def test_circle_roundtrip(circle_model, tmp_path):
    roundtrip(circle_model["GWF_1"], tmp_path)


class TestModel:
    def test_write_valid_model_without_error(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"
        model = Modflow6Model()
        # create write context
        validation_context = ValidationSettings()
        write_context = WriteContext(tmp_path)

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"

        model["dis"] = discretization_mock

        template_mock = MagicMock(spec_set=Template)
        template_mock.render.return_value = ""
        model._template = template_mock

        global_times_mock = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        status = model._write(
            model_name, global_times_mock, write_context, validation_context
        )

        # Assert.
        assert not status.has_errors()

    def test_write_without_dis_pkg_return_error(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"
        model = Modflow6Model()
        # create write context
        validation_context = ValidationSettings()
        write_context = WriteContext(tmp_path)

        template_mock = MagicMock(spec_set=Template)
        template_mock.render.return_value = ""
        model._template = template_mock

        global_times_mock = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        status = model._write(
            model_name, global_times_mock, write_context, validation_context
        )

        # Assert.
        assert status.has_errors()

    def test_write_with_invalid_pkg_returns_error(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"
        model = Modflow6Model()
        # create write context
        validation_context = ValidationSettings()
        write_context = WriteContext(tmp_path)

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"
        discretization_mock._validate.return_value = {
            "test_var": [ValidationError("error_string")]
        }

        model["dis"] = discretization_mock

        template_mock = MagicMock(spec_set=Template)
        template_mock.render.return_value = ""
        model._template = template_mock

        global_times_mock = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        status = model._write(
            model_name, global_times_mock, write_context, validation_context
        )

        # Assert.
        assert status.has_errors()

    def test_write_with_two_invalid_pkg_returns_two_errors(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"
        validation_context = ValidationSettings()
        write_context = WriteContext(simulation_directory=tmp_path)

        model = Modflow6Model()

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

        model["dis"] = discretization_mock
        model["test_package"] = package_mock

        template_mock = MagicMock(spec_set=Template)
        template_mock.render.return_value = ""
        model._template = template_mock

        global_times_mock = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        write_context = WriteContext(tmp_path)
        status = model._write(
            model_name, global_times_mock, write_context, validation_context
        )

        # Assert.
        assert len(status.errors) == 2

    @pytest.mark.parametrize("use_newton", [False, True])
    @pytest.mark.parametrize("use_under_relaxation", [False, True])
    def test_render_newton(self, use_newton, use_under_relaxation):
        # Arrange.
        model = GroundwaterFlowModel(
            newton=use_newton, under_relaxation=use_under_relaxation
        )
        write_context_mock = MagicMock(spec_set=WriteContext)

        # Act.
        output = model.render("TestModel", write_context_mock)

        # Assert.
        assert ("newton" in output) == use_newton and (
            "under_relaxation" in output
        ) == (use_newton and use_under_relaxation)


class TestGroundwaterFlowModel:
    def test_clip_box_without_state_for_boundary(self):
        # Arrange.
        state_for_boundary = None

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"
        discretization_mock.clip_box.return_value = discretization_mock

        model = GroundwaterFlowModel()
        model["dis"] = discretization_mock

        # Act.
        clipped = model.clip_box(state_for_boundary=state_for_boundary)

        # Assert.
        assert "chd_clipped" not in clipped

    @mock.patch("imod.mf6.model_gwf.create_clipped_boundary")
    def test_clip_box_with_state_for_boundary(self, create_clipped_boundary_mock):
        # Arrange.
        state_for_boundary = MagicMock(spec_set=UgridDataArray)

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"
        discretization_mock.clip_box.return_value = discretization_mock

        clipped_boundary_mock = MagicMock(spec_set=ConstantHead)
        clipped_boundary_mock.is_empty.return_value = False

        create_clipped_boundary_mock.side_effect = [
            None,
            clipped_boundary_mock,
        ]

        model = GroundwaterFlowModel()
        model["dis"] = discretization_mock

        # Act.
        clipped = model.clip_box(state_for_boundary=state_for_boundary)

        # Assert.
        assert "chd_clipped" in clipped
        create_clipped_boundary_mock.assert_called_with(
            discretization_mock["idomain"],
            state_for_boundary,
            [],
        )

    @mock.patch("imod.mf6.model_gwf.create_clipped_boundary")
    def test_clip_box_with_unassigned_boundaries_in_original_model(
        self, create_clipped_boundary_mock
    ):
        # Arrange.
        state_for_boundary = MagicMock(spec_set=UgridDataArray)

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"
        discretization_mock.is_empty.side_effect = [False, False]
        discretization_mock.clip_box.return_value = discretization_mock

        constant_head_mock = MagicMock(spec_set=ConstantHead)
        constant_head_mock.is_empty.side_effect = [False, False]
        constant_head_mock.clip_box.return_value = constant_head_mock

        unassigned_boundary_original_constant_head_mock = MagicMock(
            spec_set=ConstantHead
        )
        unassigned_boundary_original_constant_head_mock.is_empty.side_effect = [False]
        assigned_boundary_clipped_constant_head_mock = (
            unassigned_boundary_original_constant_head_mock
        )

        create_clipped_boundary_mock.side_effect = [
            unassigned_boundary_original_constant_head_mock,
            assigned_boundary_clipped_constant_head_mock,
        ]

        model = GroundwaterFlowModel()
        model["dis"] = discretization_mock
        model["chd"] = constant_head_mock

        # Act.
        clipped = model.clip_box(state_for_boundary=state_for_boundary)

        # Assert.
        assert "chd_clipped" in clipped
        create_clipped_boundary_mock.assert_called_with(
            discretization_mock["idomain"],
            state_for_boundary,
            [constant_head_mock, unassigned_boundary_original_constant_head_mock],
        )


def test_purge_empty_package(
    unstructured_flow_model: GroundwaterFlowModel,
):
    # test that purging leaves the non-empty packages in place
    original_nr_packages = len(unstructured_flow_model.items())
    unstructured_flow_model.purge_empty_packages()
    assert original_nr_packages == len(unstructured_flow_model.items())

    # test that purging removes empty packages by adding an empty well and an empty hfb
    unstructured_flow_model["wel"] = imod.mf6.Well(
        x=[],
        y=[],
        screen_top=[],
        screen_bottom=[],
        rate=[],
        minimum_k=0.0001,
    )
    geometry = gpd.GeoDataFrame(
        geometry=[shapely.linestrings([], [])],
        data={
            "resistance": [],
            "ztop": [],
            "zbottom": [],
        },
    )

    unstructured_flow_model["hfb"] = imod.mf6.HorizontalFlowBarrierResistance(geometry)
    unstructured_flow_model.purge_empty_packages()
    assert original_nr_packages == len(unstructured_flow_model.items())


def test_purge_empty_package__ignore_time(
    unstructured_flow_model: GroundwaterFlowModel,
):
    # Arrange
    rch = unstructured_flow_model["rch"]
    empty = nan_like(rch.dataset["rate"])
    transient_rate = concat([empty, rch.dataset["rate"]], dim="time")
    time_coords = [np.datetime64("2000-01-01"), np.datetime64("2001-01-01")]
    transient_rate = transient_rate.assign_coords(time=time_coords)
    rch.dataset["rate"] = transient_rate

    # Act
    original_nr_packages = len(unstructured_flow_model.items())
    unstructured_flow_model.purge_empty_packages(ignore_time=False)
    assert original_nr_packages == len(unstructured_flow_model.items())

    unstructured_flow_model.purge_empty_packages(ignore_time=True)
    assert (original_nr_packages - 1) == len(unstructured_flow_model.items())


def test_deepcopy(
    unstructured_flow_model: GroundwaterFlowModel,
):
    # test  making a deepcopy will not crash
    _ = deepcopy(unstructured_flow_model)


def test_prepare_wel_to_mf6(
    structured_flow_model: GroundwaterFlowModel,
):
    # Arrange
    # add a well to the model
    well = imod.mf6.Well(
        x=[3.0],
        y=[3.0],
        screen_top=[0.0],
        screen_bottom=[-3.0],
        rate=[1.0],
        print_flows=True,
        validate=True,
    )
    structured_flow_model["well"] = well
    # Act
    mf6_well = structured_flow_model.prepare_wel_for_mf6("well", True, True)
    # Assert
    assert isinstance(mf6_well, Mf6Wel)


def test_prepare_wel_to_mf6__error(
    structured_flow_model: GroundwaterFlowModel,
):
    # Act
    with pytest.raises(TypeError):
        structured_flow_model.prepare_wel_for_mf6("dis", True, True)


def test_get_k(structured_flow_model: GroundwaterFlowModel):
    # Act
    k = structured_flow_model._get_k()
    # Assert
    assert isinstance(k, xr.DataArray)
    np.testing.assert_allclose(k.values, 1.23)
    # Arrange
    npf = structured_flow_model.pop("npf")
    structured_flow_model["npf_other_name"] = npf
    # Act
    k = structured_flow_model._get_k()
    # Assert
    assert isinstance(k, xr.DataArray)
    np.testing.assert_allclose(k.values, 1.23)


def test_get_domain_geometry(structured_flow_model: GroundwaterFlowModel):
    # Act
    top, bottom, idomain = structured_flow_model._get_domain_geometry()
    # Assert
    assert isinstance(top, xr.DataArray)
    assert isinstance(bottom, xr.DataArray)
    assert isinstance(idomain, xr.DataArray)

    assert top.dtype == np.float64
    assert bottom.dtype == np.float64
    assert idomain.dtype == np.int32

    assert np.all(np.isin(top.values, [10.0]))
    assert np.all(np.isin(bottom.values, [-1.0, -2.0, -3.0]))
    assert np.all(np.isin(idomain.values, [2]))


def test_model_options_validation(
    structured_flow_model: GroundwaterFlowModel,
):
    # Act
    status = structured_flow_model.validate("modelname")
    # Assert
    assert not status.has_errors()

    # Arrange
    structured_flow_model._options["newton"] = 1
    # Act
    status = structured_flow_model.validate("modelname")
    # Assert
    assert status.has_errors()


def test_model_init_validation(
    structured_flow_model: GroundwaterFlowModel,
):
    # Act
    structured_flow_model.validate_init_schemata_options(
        validate=True,
    )
    # Arrange
    structured_flow_model._options["newton"] = 1
    # Act
    structured_flow_model.validate_init_schemata_options(
        validate=False,
    )
    with pytest.raises(ValidationError):
        structured_flow_model.validate_init_schemata_options(
            validate=True,
        )
