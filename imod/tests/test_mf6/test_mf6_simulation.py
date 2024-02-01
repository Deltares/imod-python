import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.model import Modflow6Model
from imod.mf6.model_gwf import GroundwaterFlowModel
from imod.mf6.multimodel.modelsplitter import PartitionInfo
from imod.mf6.simulation import get_models, get_packages
from imod.mf6.statusinfo import NestedStatusInfo, StatusInfo
from imod.schemata import ValidationError
from imod.typing.grid import zeros_like


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


@pytest.mark.usefixtures("circle_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_open_head(circle_model, tmp_path):
    simulation = circle_model

    # Should throw error when model not run yet.
    with pytest.raises(RuntimeError):
        simulation.open_head()

    modeldir = tmp_path / "circle"
    simulation.write(modeldir)
    simulation.run()

    # open heads without time conversion
    head_notime = simulation.open_head()

    assert isinstance(head_notime, xu.UgridDataArray)
    assert head_notime.dims == ("time", "layer", "mesh2d_nFaces")
    assert head_notime.shape == (1, 2, 216)

    # open heads with time conversion.
    head = simulation.open_head(
        simulation_start_time=datetime(2013, 3, 11, 22, 0, 0), time_unit="w"
    )
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (1, 2, 216)
    assert str(head.coords["time"].values[()][0]) == "2013-03-18T22:00:00.000000000"


@pytest.mark.usefixtures("circle_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_open_head_relative_path(circle_model, tmp_path):
    simulation = circle_model

    # Should throw error when model not run yet.
    with pytest.raises(RuntimeError):
        simulation.open_head()

    # Temporarily change directory to tmp_path
    with imod.util.cd(tmp_path):
        modeldir = Path(".") / "circle"
        simulation.write(modeldir)
        simulation.run()
        head = simulation.open_head()

    assert isinstance(head, xu.UgridDataArray)
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (1, 2, 216)


@pytest.mark.usefixtures("circle_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_open_flow_budget(circle_model, tmp_path):
    simulation = circle_model

    # Should throw error when model not run yet.
    with pytest.raises(RuntimeError):
        simulation.open_flow_budget()

    modeldir = tmp_path / "circle"
    simulation.write(modeldir, binary=False, use_absolute_paths=True)
    simulation.run()

    budget = simulation.open_flow_budget()

    assert isinstance(budget, dict)
    assert sorted(budget.keys()) == [
        "chd",
        "flow-horizontal-face",
        "flow-horizontal-face-x",
        "flow-horizontal-face-y",
        "flow-lower-face",
    ]
    assert isinstance(budget["chd"], xu.UgridDataArray)


@pytest.mark.usefixtures("circle_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
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


@pytest.mark.usefixtures("transient_twri_model")
@pytest.fixture(scope="function")
def split_transient_twri_model(transient_twri_model):
    active = transient_twri_model["GWF_1"].domain.sel(layer=1)

    number_partitions = 3
    split_location = np.linspace(active.y.min(), active.y.max(), number_partitions + 1)

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
            \t* Model 1:
            \t\t* Package 1:
            \t\t\t* Some error"""
        )

        pkg_status_info = StatusInfo("Package 1")
        pkg_status_info.add_error("Some error")

        model_status_info = NestedStatusInfo("Model 1")
        model_status_info.add(pkg_status_info)

        model_mock = MagicMock(spec_set=Modflow6Model)
        model_mock._model_id = "test_model_id"
        model_mock.write.return_value = model_status_info

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
    @mock.patch("imod.mf6.simulation.ExchangeCreator_Unstructured")
    def test_split_multiple_models(
        self,
        exchange_creator_unstructured_mock,
        slice_model_mock,
        circle_dis,
        setup_simulation,
    ):
        # Arrange.
        idomain, _, _ = circle_dis

        simulation = setup_simulation

        model_mock1 = MagicMock(spec_set=GroundwaterFlowModel)
        model_mock1._model_id = "test_model_id1"

        model_mock2 = MagicMock(spec_set=GroundwaterFlowModel)
        model_mock2._model_id = "test_model_id2"

        simulation["test_model1"] = model_mock1
        simulation["test_model2"] = model_mock2

        simulation["solver"].dataset = xr.Dataset(
            {"modelnames": ["test_model1", "test_model2"]}
        )

        slice_model_mock.return_value = MagicMock(spec_set=GroundwaterFlowModel)

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

        active_domain1 = submodel_labels.where(submodel_labels == 0, 0).where(submodel_labels != 0, 1)
        active_domain2 = submodel_labels.where(submodel_labels == 1, 0).where(submodel_labels != 1, 1)
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

    @mock.patch("imod.mf6.simulation.slice_model", autospec=True)
    @mock.patch("imod.mf6.simulation.ExchangeCreator_Structured", autospec=True)
    @mock.patch("imod.mf6.simulation.create_partition_info")
    def test_split_multiple_models_creates_expected_number_of_exchanges(
        self,
        create_partition_info_mock,
        exchange_creator_mock,
        slice_model_mock,
        basic_dis,
        setup_simulation,
    ):
        # Arrange.
        idomain, top, bottom = basic_dis

        simulation = setup_simulation

        model_mock1 = MagicMock(spec_set=GroundwaterFlowModel)
        model_mock1._model_id = "test_model_id1"
        model_mock1.domain = idomain

        model_mock2 = MagicMock(spec_set=GroundwaterFlowModel)
        model_mock2._model_id = "test_model_id2"
        model_mock2.domain = idomain

        simulation["test_model1"] = model_mock1
        simulation["test_model2"] = model_mock2

        simulation["solver"].dataset = xr.Dataset(
            {"modelnames": ["test_model1", "test_model2"]}
        )

        slice_model_mock.return_value = MagicMock(spec_set=GroundwaterFlowModel)

        active = idomain.sel(layer=1)
        submodel_labels = xr.zeros_like(active).where(active.y > 50, 1)

        create_partition_info_mock.return_value = [
            PartitionInfo(id=0, active_domain=xr.DataArray(0)),
            PartitionInfo(id=1, active_domain=xr.DataArray(1)),
        ]
        # Act.
        _ = simulation.split(submodel_labels)

        # Assert.
        exchange_creator_mock.assert_called_with(
            submodel_labels, create_partition_info_mock()
        )

        # fmt: off
        assert exchange_creator_mock.return_value.create_gwfgwf_exchanges.call_count == 2  # noqa: E501
        call1 = exchange_creator_mock.return_value.create_gwfgwf_exchanges.call_args_list[0][0]  # noqa: E501
        call2 = exchange_creator_mock.return_value.create_gwfgwf_exchanges.call_args_list[1][0]  # noqa: E501
        # fmt: on

        assert call1[0] == "test_model1"
        xr.testing.assert_equal(call1[1], idomain.layer)

        assert call2[0] == "test_model2"
        xr.testing.assert_equal(call2[1], idomain.layer)

    @pytest.mark.usefixtures("transient_twri_model")
    def test_exchanges_in_simulation_file(self, transient_twri_model, tmp_path):
        # Arrange
        active = transient_twri_model["GWF_1"].domain.sel(layer=1)
        number_partitions = 3
        split_location = np.linspace(
            active.y.min(), active.y.max(), number_partitions + 1
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

    @pytest.mark.usefixtures("transient_twri_model")
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

    @pytest.mark.usefixtures("split_transient_twri_model")
    def test_prevent_split_after_split(
        self,
        split_transient_twri_model,
    ):
        # Arrange.
        split_simulation = split_transient_twri_model

        # Act/Assert
        with pytest.raises(RuntimeError):
            _ = split_simulation.split(None)

    @pytest.mark.usefixtures("split_transient_twri_model")
    def test_prevent_clip_box_after_split(
        self,
        split_transient_twri_model,
    ):
        # Arrange.
        split_simulation = split_transient_twri_model

        # Act/Assert
        with pytest.raises(RuntimeError):
            _ = split_simulation.clip_box()

    @pytest.mark.usefixtures("split_transient_twri_model")
    def test_prevent_regrid_like_after_split(
        self,
        split_transient_twri_model,
    ):
        # Arrange.
        split_simulation = split_transient_twri_model

        # Act/Assert
        with pytest.raises(RuntimeError):
            _ = split_simulation.regrid_like(None, None)


def compare_submodel_partition_info(first: PartitionInfo, second: PartitionInfo):
    return (first.id == second.id) and np.array_equal(
        first.active_domain, second.active_domain
    )
