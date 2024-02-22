from copy import deepcopy
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr
import xugrid as xu
from jinja2 import Template
from xugrid.core.wrap import UgridDataArray

import imod
from imod.mf6 import ConstantHead
from imod.mf6.model import Modflow6Model
from imod.mf6.model_gwf import GroundwaterFlowModel
from imod.mf6.package import Package
from imod.mf6.write_context import WriteContext
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
        model = Modflow6Model()
        # create write context
        write_context = WriteContext(tmp_path)

        discretization_mock = MagicMock(spec_set=Package)
        discretization_mock._pkg_id = "dis"

        model["dis"] = discretization_mock

        template_mock = MagicMock(spec_set=Template)
        template_mock.render.return_value = ""
        model._template = template_mock

        global_times_mock = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        status = model.write(model_name, global_times_mock, True, write_context)

        # Assert.
        assert not status.has_errors()

    def test_write_without_dis_pkg_return_error(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"
        model = Modflow6Model()
        # create write context
        write_context = WriteContext(tmp_path)

        template_mock = MagicMock(spec_set=Template)
        template_mock.render.return_value = ""
        model._template = template_mock

        global_times_mock = MagicMock(spec_set=imod.mf6.TimeDiscretization)

        # Act.
        status = model.write(model_name, global_times_mock, True, write_context)

        # Assert.
        assert status.has_errors()

    def test_write_with_invalid_pkg_returns_error(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"
        model = Modflow6Model()
        # create write context
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
        status = model.write(
            model_name, global_times_mock, True, write_context=write_context
        )

        # Assert.
        assert status.has_errors()

    def test_write_with_two_invalid_pkg_returns_two_errors(self, tmpdir_factory):
        # Arrange.
        tmp_path = tmpdir_factory.mktemp("TestSimulation")
        model_name = "Test model"
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
        status = model.write(model_name, global_times_mock, True, write_context)

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
    tmp_path: Path,
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

@pytest.mark.parametrize("layer_mask", [[1,1,0], [0,1,1], [1,0,1]])
def test_mask_with_layer_array(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
    layer_mask: list[int]
):  
    """
    Specifying a mask as a dataset with only a layer coordinate is not necessarily something we want
    to maintain forever, so we should discourage external users from using this. 
    """
    nlayer = 3
    layer = np.arange(nlayer, dtype=int) + 1
    grid = unstructured_flow_model.domain.ugrid.grid
    mask =  xu.UgridDataArray(
        xr.DataArray(
            coords={"layer": layer},
            dims=["layer"],
        ),
        grid=grid,
    )
    mask.values = layer_mask

    unstructured_flow_model.mask_all_packages(mask)

    assert_model_can_run( unstructured_flow_model, "disv", tmp_path )

@pytest.mark.parametrize("mask_cells",[ [(0, 2,1)],    # case 1: disable a chd cell 
                         [(0, 3,2),(1,3,2), (2,3,2)]]) # case 2: disable all the cells the well ends up in
@pytest.mark.parametrize("inactivity_marker",[ 0, -1]) # 0 = inactive, -1 = vertical passthrough
def test_mask_structured(tmp_path: Path, structured_flow_model: GroundwaterFlowModel, mask_cells: list[tuple[int, int, int]], inactivity_marker: int):

    # Arrange 
    # add a well to the model
    well = imod.mf6.Well(
        x=[ 3.0],
        y=[ 3.0],
        screen_top=[0.0],
        screen_bottom=[ -3.0],
        rate=[1.0],
        print_flows=True,
        validate=True,
    )
    structured_flow_model["well"] = well
    cell_count = len(structured_flow_model.domain.x)*len(structured_flow_model.domain.y)*len(structured_flow_model.domain.layer)

    mask = deepcopy(structured_flow_model.domain)
    for cell in mask_cells:
        mask.values[*cell ] = inactivity_marker

    # Act
    structured_flow_model.mask_all_packages( mask)

    # Assert
    unique, counts = np.unique(structured_flow_model.domain.values.reshape(cell_count), return_counts=True)
    assert unique[0] == inactivity_marker
    assert counts[0] == len(mask_cells)
    assert counts[1] == cell_count - len(mask_cells)
    assert_model_can_run( structured_flow_model, "dis", tmp_path )

@pytest.mark.parametrize("mask_cells", [( 2,1),  # case 1: disable a chd cell. These are indices, NOT coordinates. 
                                        ( 3,2)]) # case 2: disable all the cells the well ends up in
def test_mask_structured_xy_masks_across_all_layers(tmp_path: Path, structured_flow_model: GroundwaterFlowModel, mask_cells:tuple[int, int]):
    # Arrange 
    # add a well to the model
    well = imod.mf6.Well(
        x=[ 3.0],
        y=[ 3.0],
        screen_top=[0.0],
        screen_bottom=[ -3.0],
        rate=[1.0],
        print_flows=True,
        validate=True,
    )
    structured_flow_model["well"] = well
    
    mask = deepcopy(structured_flow_model.domain.sel(layer=1))
    mask = mask.drop_vars("layer")
    mask.values[*mask_cells] = 0

    cell_count = len(structured_flow_model.domain.x)*len(structured_flow_model.domain.y)*len(structured_flow_model.domain.layer)
    
    # Act    
    structured_flow_model.mask_all_packages( mask)
    
    # Assert
    assert all(structured_flow_model.domain.isel(y = mask_cells[0], x = mask_cells[1]).values == np.zeros(len(structured_flow_model.domain.layer)))
    unique, counts = np.unique(structured_flow_model.domain.values.reshape(cell_count), return_counts=True)
    assert counts[0] == len(structured_flow_model.domain.layer)
    assert counts[1] == cell_count - len(structured_flow_model.domain.layer)
    assert_model_can_run( structured_flow_model, "dis", tmp_path )    


@pytest.mark.parametrize("mask_cell", [[1,1], [1,33]])
@pytest.mark.parametrize("inactivity_marker",[ 0, -1])  # 0 = inactive, -1 = vertical passthrough
def test_mask_unstructured(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
    mask_cell: list[int],
    inactivity_marker: int
):  

    # Arrange     
    layer_dim = len(unstructured_flow_model.domain.coords["layer"].values)
    planar_dim = len(unstructured_flow_model.domain.coords["mesh2d_nFaces"].values)
    cell_count = planar_dim * layer_dim
    mask = deepcopy(unstructured_flow_model.domain)
    mask.values[*mask_cell] = inactivity_marker

    # Act  
    unstructured_flow_model.mask_all_packages(mask)

    # Assert
    unique, counts = np.unique(unstructured_flow_model.domain.values.reshape(cell_count), return_counts=True)
    assert unstructured_flow_model.domain.values[*mask_cell] == inactivity_marker
    assert unique[0] == inactivity_marker
    assert counts[0] == 1
    assert counts[1] == cell_count - 1
    assert_model_can_run( unstructured_flow_model, "disv", tmp_path )


def test_mask_with_time_coordinate(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
):      

    nlayer = 3
    layer = np.arange(nlayer, dtype=int) + 1
    grid = unstructured_flow_model.domain.ugrid.grid
    mask =  xu.UgridDataArray(
        xr.DataArray(
            coords={"layer": layer, "time" : [1,2] },
            dims=["layer", "time"],
        ),
        grid=grid,
    )
    mask.sel(time=1).values = np.array([1,1,0])
    mask.sel(time=2).values = np.array([1,0,1])

    with pytest.raises(ValueError):
        unstructured_flow_model.mask_all_packages(mask)

def test_mask_everything(
    tmp_path: Path,
    unstructured_flow_model: GroundwaterFlowModel,
):      
    mask = deepcopy(unstructured_flow_model.domain)
    mask.values[:,:] = -1

    with pytest.raises(ValueError):
        unstructured_flow_model.mask_all_packages(mask)    