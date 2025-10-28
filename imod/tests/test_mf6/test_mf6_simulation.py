import os
import re
import sys
import textwrap
from copy import deepcopy
from datetime import datetime
from filecmp import dircmp
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import rasterio
import tomli
import tomli_w
import xarray as xr
import xugrid as xu
from pytest_cases import parametrize_with_cases

import imod
from imod.common.statusinfo import NestedStatusInfo, StatusInfo
from imod.common.utilities.version import get_version
from imod.logging import LoggerType, LogLevel
from imod.mf6 import LayeredWell, Well
from imod.mf6.model import Modflow6Model
from imod.mf6.multimodel.modelsplitter import PartitionInfo
from imod.mf6.oc import OutputControl
from imod.mf6.regrid.regrid_schemes import (
    DiscretizationRegridMethod,
    NodePropertyFlowRegridMethod,
    StorageCoefficientRegridMethod,
)
from imod.mf6.simulation import Modflow6Simulation
from imod.prepare.topsystem.default_allocation_methods import (
    SimulationAllocationOptions,
    SimulationDistributingOptions,
)
from imod.schemata import ValidationError
from imod.tests.fixtures.mf6_small_models_fixture import (
    grid_data_structured,
)
from imod.typing.grid import zeros_like


def roundtrip(simulation, tmpdir_factory, name):
    # TODO: look at the values?
    tmp_path = tmpdir_factory.mktemp(name)
    simulation.dump(tmp_path)
    back = imod.mf6.Modflow6Simulation.from_file(tmp_path / f"{simulation.name}.toml")
    assert isinstance(back, imod.mf6.Modflow6Simulation)


def test_twri_roundtrip(twri_model, tmpdir_factory):
    roundtrip(twri_model, tmpdir_factory, "twri")


def test_twri_hfb_roundtrip(twri_model_hfb, tmpdir_factory):
    roundtrip(twri_model_hfb, tmpdir_factory, "twri")


def test_twri_transient_roundtrip(transient_twri_model, tmpdir_factory):
    roundtrip(transient_twri_model, tmpdir_factory, "twri_transient")


def test_twri_disv_roundtrip(twri_disv_model, tmpdir_factory):
    roundtrip(twri_disv_model, tmpdir_factory, "twri_disv")


def test_circle_roundtrip(circle_model, tmpdir_factory):
    roundtrip(circle_model, tmpdir_factory, "circle")


def test_dump_version_number__version_written(twri_model, tmpdir_factory):
    # Arrange
    tmp_path = tmpdir_factory.mktemp("twri")
    # Act
    twri_model.dump(tmp_path)
    # Assert
    toml_path = tmp_path / f"{twri_model.name}.toml"
    with open(toml_path, "rb") as f:
        toml_content = tomli.load(f)
    assert toml_content["version"]["imod-python"] == get_version()


def test_from_file_version_logged__version_in_dumped(twri_model, tmpdir_factory):
    """
    Tested if a warning is thrown when there is a mismatch between version
    numbers of saved model and current iMOD Python version.
    """
    # Arrange
    tmp_path = tmpdir_factory.mktemp("twri")
    twri_model.dump(tmp_path)
    toml_path = tmp_path / f"{twri_model.name}.toml"
    with open(toml_path, "rb") as f:
        toml_content = tomli.load(f)
    toml_content["version"]["imod-python"] = "0.0.0"

    toml_path_adapted = tmp_path / f"{twri_model.name}_adapted.toml"
    with open(toml_path_adapted, "wb") as f:
        tomli_w.dump(toml_content, f)
    # Act
    logfile_path = tmp_path / "logfile.txt"
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        imod.mf6.Modflow6Simulation.from_file(toml_path_adapted)

    # Assert
    with open(logfile_path, "r") as log_file:
        log = log_file.read()
        assert f"iMOD Python version in current environment: {imod.__version__}" in log
        assert "iMOD Python version in dumped simulation: 0.0.0" in log


def test_from_file_version_logged__no_version_in_dumped(twri_model, tmpdir_factory):
    """
    Tested if a warning is thrown when there is a mismatch between version
    numbers of saved model and current iMOD Python version.
    """
    # Arrange
    tmp_path = tmpdir_factory.mktemp("twri")
    twri_model.dump(tmp_path)
    toml_path = tmp_path / f"{twri_model.name}.toml"
    with open(toml_path, "rb") as f:
        toml_content = tomli.load(f)
    toml_content.pop("version")

    toml_path_adapted = tmp_path / f"{twri_model.name}_adapted.toml"
    with open(toml_path_adapted, "wb") as f:
        tomli_w.dump(toml_content, f)
    # Act
    logfile_path = tmp_path / "logfile.txt"
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        imod.mf6.Modflow6Simulation.from_file(toml_path_adapted)

    # Assert
    with open(logfile_path, "r") as log_file:
        log = log_file.read()
        assert f"iMOD Python version in current environment: {imod.__version__}" in log
        assert "No iMOD Python version information found in dumped simulation." in log


def test_twri_gdal(twri_model, tmpdir_factory):
    """
    Test of dumping a structured model with a CRS results in the necessary
    attributes for GDAL and a parsable CRS.
    """
    tmp_path = tmpdir_factory.mktemp("twri")
    twri_model.dump(tmp_path, crs="EPSG:28992")
    ds = xr.open_dataset(tmp_path / "GWF_1" / "dis.nc")
    assert ds.coords["x"].attrs["axis"] == "X"
    assert ds.coords["y"].attrs["axis"] == "Y"
    # Convert long string with full description to EPSG code
    crs = rasterio.CRS.from_string(ds["spatial_ref"].attrs["spatial_ref"])
    assert crs == "EPSG:28992"


def test_twri_disv_mdal_compliant_semi_roundtrip(twri_disv_model, tmpdir_factory):
    """
    Test if dumping an unstructured model with mdal_compliant=True also results
    in a package written with a crs and unstacked layers, which can be stacked.
    """
    tmp_path = tmpdir_factory.mktemp("twri")
    twri_disv_model.dump(tmp_path, crs="EPSG:28992", mdal_compliant=True)
    ds = xu.open_dataset(tmp_path / "GWF_1" / "dis.nc")
    assert len(ds.data_vars) == 7
    # Convert long string with full description to EPSG code
    crs = rasterio.CRS.from_string(ds.coords["spatial_ref"].attrs["spatial_ref"])
    assert crs == "EPSG:28992"
    # Test if running from_mdal_compliant stacks variables properly again.
    ds = imod.util.spatial.from_mdal_compliant_ugrid2d(ds)
    assert len(ds.data_vars) == 3


def test_simulation_open_head(circle_model, tmp_path):
    simulation = circle_model

    # Should throw error when model not run yet.
    with pytest.raises(RuntimeError):
        simulation.open_head()

    modeldir = tmp_path / "circle"
    simulation.write(modeldir)
    simulation.run()
    head = simulation.open_head()

    assert isinstance(head, xu.UgridDataArray)
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (52, 2, 216)

    # open heads with time conversion.
    head = simulation.open_head(
        simulation_start_time=datetime(2013, 3, 11, 22, 0, 0), time_unit="w"
    )
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (52, 2, 216)
    assert str(head.coords["time"].values[()][0]) == "2013-04-29T22:00:00.000000000"


class PathCases:
    def case_absolute_path(self, tmp_path):
        return tmp_path.resolve() / "absolute"

    def case_relative_path(self):
        return Path(".") / "relative"

    def case_space_path(self, tmp_path):
        return tmp_path / "dir with spaces"


@parametrize_with_cases("path", cases=PathCases)
def test_simulation_write_run_open__different_paths(circle_model, tmp_path, path):
    simulation = circle_model

    # Temporarily change directory to tmp_path (for relative path)
    with imod.util.cd(tmp_path):
        path.mkdir(parents=True, exist_ok=True)
        simulation.write(path)
        simulation.run()
        head = simulation.open_head()
        # Assert not an empty array is returned
        assert isinstance(head, xu.UgridDataArray)
        assert head.shape == (52, 2, 216)


def test_simulation_open_flow_budget(circle_model, tmp_path):
    simulation = circle_model

    # Should throw error when model not run yet.
    with pytest.raises(RuntimeError):
        simulation.open_flow_budget()

    modeldir = tmp_path / "circle"
    simulation.write(modeldir, binary=False, use_absolute_paths=True)
    simulation.run()

    budget = simulation.open_flow_budget()

    assert isinstance(budget, xu.UgridDataset)
    assert sorted(budget.keys()) == [
        "chd_chd",
        "flow-horizontal-face",
        "flow-horizontal-face-x",
        "flow-horizontal-face-y",
        "flow-lower-face",
    ]
    assert isinstance(budget["chd_chd"], xu.UgridDataArray)


def test_write_circle_model_twice(circle_model, tmp_path):
    simulation = circle_model

    # write simulation, then write the simulation a second time
    simulation.write(tmp_path / "first_time", binary=False)
    simulation.write(tmp_path / "second_time", binary=False)

    # check that text output is the same
    diff = dircmp(tmp_path / "first_time", tmp_path / "second_time")
    assert len(diff.diff_files) == 0
    assert len(diff.left_only) == 0
    assert len(diff.right_only) == 0


def test_simulation_clip_box__validation_settings_preserved(circle_model):
    simulation = circle_model
    simulation.set_validation_settings(
        imod.mf6.ValidationSettings(strict_hfb_validation=False)
    )
    clipped_simulation = simulation.clip_box(y_min=-50, y_max=0)
    assert simulation._validation_context == clipped_simulation._validation_context


def test_simulation_open_concentration_fail(circle_model, tmp_path):
    """No transport model is assigned, so should throw error when opening concentrations"""
    simulation = circle_model

    modeldir = tmp_path / "circle"
    simulation.write(modeldir, binary=False, use_absolute_paths=True)
    simulation.run()

    with pytest.raises(ValueError):
        simulation.open_concentration()


@pytest.fixture(scope="function")
def sample_gwfgwf_structured():
    ds = xr.Dataset()
    ds["cell_id1"] = xr.DataArray(
        [[1, 1], [2, 1], [3, 1]],
        dims=("index", "cell_dims1"),
        coords={"cell_dims1": ["row_1", "column_1"]},
    )
    ds["cell_id2"] = xr.DataArray(
        [[1, 2], [2, 2], [3, 2]],
        dims=("index", "cell_dims2"),
        coords={"cell_dims2": ["row_2", "column_2"]},
    )
    ds["layer"] = xr.DataArray([12, 13, 14], dims="layer")
    ds["cl1"] = xr.DataArray(np.ones(3), dims="index")
    ds["cl2"] = xr.DataArray(np.ones(3), dims="index")
    ds["hwva"] = ds["cl1"] + ds["cl2"]

    ds = ds.stack(cell_id=("layer", "index"), create_index=False).reset_coords()
    ds["cell_id1"] = ds["cell_id1"].T
    ds["cell_id2"] = ds["cell_id2"].T

    return imod.mf6.GWFGWF("name1", "name2", **ds)


@pytest.fixture(scope="function")
def setup_simulation():
    simulation = imod.mf6.Modflow6Simulation("TestSimulation")
    simulation["time_discretization"] = MagicMock(spec_set=imod.mf6.TimeDiscretization)

    solver_mock = MagicMock(spec=imod.mf6.Solution)
    solver_mock.__getitem__.side_effect = lambda x: getattr(solver_mock, x)
    solver_mock.modelnames = []

    simulation["solver"] = solver_mock

    return simulation


@pytest.fixture(scope="function")
def split_transient_twri_model(transient_twri_model):
    active = transient_twri_model["GWF_1"].domain.sel(layer=1)

    number_partitions = 3
    split_location = np.linspace(
        active.y.min().item(), active.y.max().item(), number_partitions + 1
    )

    coords = active.coords
    submodel_labels = zeros_like(active)
    for id in np.arange(1, number_partitions):
        submodel_labels.loc[
            (coords["y"] > split_location[id]) & (coords["y"] <= split_location[id + 1])
        ] = id

    split_simulation = transient_twri_model.split(submodel_labels)

    return split_simulation


class TestModflow6Simulation:
    def test_write_sets_directory(self, tmp_path, setup_simulation):
        # Arrange.
        simulation = setup_simulation

        # Assert
        # Should be None upon initialization
        assert simulation.directory is None

        # Act.
        simulation.write(tmp_path)

        # Assert.
        assert simulation.directory is not None
        assert simulation.directory == tmp_path

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

        expected_text = textwrap.dedent(
            """
            Simulation validation status:
                - Model 1:
                    - Package 1:
                        - Some error"""
        )

        pkg_status_info = StatusInfo("Package 1")
        pkg_status_info.add_error("Some error")

        model_status_info = NestedStatusInfo("Model 1")
        model_status_info.add(pkg_status_info)

        model_mock = MagicMock(spec_set=Modflow6Model)
        model_mock._model_id = "test_model_id"
        model_mock._write.return_value = model_status_info

        simulation["test_model"] = model_mock

        # Act/Assert.
        with pytest.raises(ValidationError, match=re.escape(expected_text)):
            simulation.write(tmp_path)

    @mock.patch("imod.mf6.simulation.ExchangeCreator_Unstructured")
    def test_split_simulation_only_has_packages(
        self, exchange_creator_mock, basic_unstructured_dis, setup_simulation
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
        with pytest.raises(ValueError):
            _ = simulation.split(submodel_labels)

    def test_split_multiple_flow_models(self, structured_flow_simulation_2_flow_models):
        # Arrange.
        active = structured_flow_simulation_2_flow_models["flow"].domain.sel(layer=1)
        submodel_labels = xr.zeros_like(active)
        submodel_labels.values[:, 3:] = 1

        # Act
        with pytest.raises(ValueError):
            _ = structured_flow_simulation_2_flow_models.split(submodel_labels)

    def test_regrid_multiple_flow_models(
        self, structured_flow_simulation_2_flow_models
    ):
        # Arrange
        finer_idomain = grid_data_structured(np.int32, 1, 0.4)

        # Act
        with pytest.raises(ValueError):
            _ = structured_flow_simulation_2_flow_models.regrid_like(
                "regridded_model", finer_idomain
            )

    def test_clip_multiple_flow_models(self, structured_flow_simulation_2_flow_models):
        # Arrange
        active = structured_flow_simulation_2_flow_models["flow"].domain
        grid_y_min = active.coords["y"].values[-1]
        grid_y_max = active.coords["y"].values[0]

        # Act/Assert
        with pytest.raises(ValueError):
            _ = structured_flow_simulation_2_flow_models.clip_box(
                y_min=grid_y_min, y_max=grid_y_max / 2
            )

    def test_exchanges_in_simulation_file(self, transient_twri_model, tmp_path):
        # Arrange
        active = transient_twri_model["GWF_1"].domain.sel(layer=1)
        number_partitions = 3
        split_location = np.linspace(
            active.y.min().item(), active.y.max().item(), number_partitions + 1
        )

        coords = active.coords
        submodel_labels = zeros_like(active)
        for id in np.arange(1, number_partitions):
            submodel_labels.loc[
                (coords["y"] > split_location[id])
                & (coords["y"] <= split_location[id + 1])
            ] = id

        # Act
        split_simulation = transient_twri_model.split(submodel_labels)

        # Assert
        assert len(split_simulation["split_exchanges"]) == 2
        split_simulation.write(tmp_path, False, True, False)

        expected_exchanges_block = textwrap.dedent(
            """\
            exchanges
              GWF6-GWF6 GWF_1_0_GWF_1_1.gwfgwf GWF_1_0 GWF_1_1
              GWF6-GWF6 GWF_1_1_GWF_1_2.gwfgwf GWF_1_1 GWF_1_2

            end exchanges
            """
        )
        with open(tmp_path / "mfsim.nam", mode="r") as mfsim_nam:
            namfile_content = mfsim_nam.read()
        assert expected_exchanges_block in namfile_content

    def test_write_exchanges(
        self, transient_twri_model, sample_gwfgwf_structured, tmp_path
    ):
        # Arrange
        transient_twri_model["split_exchanges"] = [sample_gwfgwf_structured]

        # Act
        transient_twri_model.write(tmp_path, True, True, True)

        # Assert
        _, filename, _, _ = sample_gwfgwf_structured.get_specification()
        assert Path.exists(tmp_path / filename)

    def test_prevent_split_after_split(
        self,
        split_transient_twri_model,
    ):
        # Arrange.
        split_simulation = split_transient_twri_model

        # Act/Assert
        with pytest.raises(RuntimeError):
            _ = split_simulation.split(None)

    def test_prevent_clip_box_after_split(
        self,
        split_transient_twri_model,
    ):
        # Arrange.
        split_simulation = split_transient_twri_model

        # Act/Assert
        with pytest.raises(RuntimeError):
            _ = split_simulation.clip_box()

    def test_deepcopy(split_transient_twri_model):
        # test  making a deepcopy will not crash
        _ = deepcopy(split_transient_twri_model)

    def test_prevent_regrid_like_after_split(
        self,
        split_transient_twri_model,
    ):
        # Arrange.
        split_simulation = split_transient_twri_model

        # Act/Assert
        with pytest.raises(RuntimeError):
            _ = split_simulation.regrid_like(
                "new_simulation", split_transient_twri_model["GWF_1_2"].domain
            )


def compare_submodel_partition_info(first: PartitionInfo, second: PartitionInfo):
    return (first.id == second.id) and np.array_equal(
        first.active_domain, second.active_domain
    )


@pytest.mark.unittest_jit
def test_import_from_imod5(imod5_dataset, tmp_path):
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]

    datelist = pd.date_range(start="1/1/1989", end="1/1/2013", freq="W")

    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        datelist,
        SimulationAllocationOptions,
        SimulationDistributingOptions,
    )
    simulation["imported_model"]["oc"] = OutputControl(
        save_head="last", save_budget="last"
    )
    simulation.create_time_discretization(["01-01-2003", "02-01-2003"])
    # Cleanup
    # Remove HFB packages outside domain
    # TODO: Build in support for hfb packages outside domain
    for hfb_outside in ["hfb-24", "hfb-26"]:
        simulation["imported_model"].pop(hfb_outside)
    # Align NoData to domain
    idomain = simulation["imported_model"].domain
    simulation.mask_all_models(idomain)
    # write and validate the simulation.
    simulation.write(tmp_path, binary=False, validate=True)

    # Test if simulation attribute appropiately set
    assert simulation._validation_context.strict_well_validation is False


@pytest.mark.unittest_jit
def test_from_imod5__has_cap_data(imod5_dataset):
    imod5_data = deepcopy(imod5_dataset[0])
    period_data = imod5_dataset[1]

    imod5_data["cap"] = {}
    msw_bound = imod5_data["bnd"]["ibound"].isel(layer=0, drop=True)
    imod5_data["cap"]["boundary"] = msw_bound
    imod5_data["cap"]["wetted_area"] = xr.ones_like(msw_bound) * 100
    imod5_data["cap"]["urban_area"] = xr.ones_like(msw_bound) * 200
    imod5_data["cap"]["artificial_recharge"] = msw_bound
    imod5_data["cap"]["artificial_recharge_layer"] = xr.ones_like(msw_bound) + 1
    imod5_data["cap"]["artificial_recharge_capacity"] = xr.DataArray(25.0)

    datelist = pd.date_range(start="1/1/1989", end="1/1/2013", freq="W")

    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        datelist,
        SimulationAllocationOptions,
        SimulationDistributingOptions,
    )

    gwf_model = simulation["imported_model"]

    assert "msw-rch" in gwf_model.keys()
    assert "msw-sprinkling" in gwf_model.keys()


@pytest.mark.unittest_jit
def test_from_imod5__strict_well_validation_set(imod5_dataset):
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]

    datelist = pd.date_range(start="1/1/1989", end="1/1/1990", freq="W")

    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        datelist,
        SimulationAllocationOptions,
        SimulationDistributingOptions,
    )
    assert simulation._validation_context.strict_well_validation is False
    assert Modflow6Simulation("test")._validation_context.strict_well_validation is True


@pytest.mark.unittest_jit
def test_import_from_imod5__correct_well_type(imod5_dataset):
    # Unpack
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]
    # Temporarily change layer number to 0, to force Well object instead of
    # LayeredWell
    original_wel_layer = imod5_data["wel-WELLS_L3"]["layer"]
    imod5_data["wel-WELLS_L3"]["layer"] = [0] * len(original_wel_layer)
    # Other arrangement
    default_simulation_allocation_options = SimulationAllocationOptions
    default_simulation_distributing_options = SimulationDistributingOptions
    datelist = pd.date_range(start="1/1/1989", end="1/1/2013", freq="W")

    # Act
    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        datelist,
        default_simulation_allocation_options,
        default_simulation_distributing_options,
    )
    # Set layer back to right value (before AssertionError might be thrown)
    imod5_data["wel-WELLS_L3"]["layer"] = original_wel_layer
    # Assert
    assert isinstance(simulation["imported_model"]["wel-WELLS_L3"], Well)
    assert isinstance(simulation["imported_model"]["wel-WELLS_L4"], LayeredWell)
    assert isinstance(simulation["imported_model"]["wel-WELLS_L5"], LayeredWell)


@pytest.mark.unittest_jit
def test_import_from_imod5__well_steady_state(imod5_dataset):
    # Unpack
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]

    sto = imod5_data.pop("sto")

    # Other arrangement
    default_simulation_allocation_options = SimulationAllocationOptions
    default_simulation_distributing_options = SimulationDistributingOptions
    datelist = pd.date_range(start="1/1/1989", end="1/1/2013", freq="W")

    # Act
    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        datelist,
        default_simulation_allocation_options,
        default_simulation_distributing_options,
    )
    # Assert
    gwf = simulation["imported_model"]
    assert "time" not in gwf["wel-WELLS_L3"].dataset.coords
    assert "time" not in gwf["wel-WELLS_L4"].dataset.coords
    assert "time" not in gwf["wel-WELLS_L5"].dataset.coords
    # Teardown
    # Reassign storage package again
    imod5_data["sto"] = sto


@pytest.mark.unittest_jit
def test_import_from_imod5__nonstandard_regridding(imod5_dataset, tmp_path):
    imod5_data = imod5_dataset[0]
    period_data = imod5_dataset[1]

    regridding_option = {}
    regridding_option["npf"] = NodePropertyFlowRegridMethod()
    regridding_option["dis"] = DiscretizationRegridMethod()
    regridding_option["sto"] = StorageCoefficientRegridMethod()
    times = pd.date_range(start="1/1/2018", end="12/1/2018", freq="ME")

    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        times,
        SimulationAllocationOptions,
        SimulationDistributingOptions,
        regridding_option,
    )
    simulation["imported_model"]["oc"] = OutputControl(
        save_head="last", save_budget="last"
    )
    simulation.create_time_discretization(["01-01-2003", "02-01-2003"])
    # Cleanup
    # Remove HFB packages outside domain
    # TODO: Build in support for hfb packages outside domain
    for hfb_outside in ["hfb-24", "hfb-26"]:
        simulation["imported_model"].pop(hfb_outside)
    # Align NoData to domain
    idomain = simulation["imported_model"].domain
    simulation.mask_all_models(idomain)
    # write and validate the simulation.
    simulation.write(tmp_path, binary=False, validate=True)


@pytest.mark.unittest_jit
def test_import_from_imod5_no_storage_no_recharge(imod5_dataset, tmp_path):
    # this test imports an imod5 simulation, but it has no recharge and no storage package.
    imod5_data = imod5_dataset[0]
    imod5_data.pop("sto")
    imod5_data.pop("rch")
    period_data = imod5_dataset[1]

    times = pd.date_range(start="1/1/2018", end="12/1/2018", freq="ME")

    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        times,
        SimulationAllocationOptions,
        SimulationDistributingOptions,
    )
    simulation["imported_model"]["oc"] = OutputControl(
        save_head="last", save_budget="last"
    )
    simulation.create_time_discretization(["01-01-2003", "02-01-2003"])
    # Cleanup
    # Remove HFB packages outside domain
    # TODO: Build in support for hfb packages outside domain
    for hfb_outside in ["hfb-24", "hfb-26"]:
        simulation["imported_model"].pop(hfb_outside)
    # check storage is present and rch is absent
    assert not simulation["imported_model"]["sto"].dataset["transient"].values[()]
    package_keys = simulation["imported_model"].keys()
    for key in package_keys:
        assert key[0:3] != "rch"
    # Align NoData to domain
    idomain = simulation["imported_model"].domain
    simulation.mask_all_models(idomain)
    # write and validate the simulation.
    simulation.write(tmp_path, binary=False, validate=True)
