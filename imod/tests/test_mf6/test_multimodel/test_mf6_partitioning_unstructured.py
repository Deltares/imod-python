from filecmp import dircmp
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely
import xugrid as xu
from pytest_cases import parametrize_with_cases

import imod
from imod.mf6 import Modflow6Simulation
from imod.typing.grid import zeros_like

@pytest.mark.usefixtures("circle_model")
@pytest.fixture(scope="function")
def idomain_top(circle_model):
    idomain = circle_model["GWF_1"].domain
    return idomain.isel(layer=0)


class PartitionArrayCases:
    def case_two_parts(self, idomain_top) -> xu.UgridDataArray:
        two_parts = zeros_like(idomain_top)
        two_parts.values[108:] = 1
        two_parts["name"] = "two_parts"
        return two_parts

    def case_three_parts(self, idomain_top) -> xu.UgridDataArray:
        three_parts = zeros_like(idomain_top)
        three_parts.values[72:144] = 1
        three_parts.values[144:] = 2
        three_parts["name"] = "three_parts"

        return three_parts

    def case_concentric(self, idomain_top) -> xu.UgridDataArray:
        centroids = idomain_top.ugrid.grid.centroids
        dist = np.sqrt(
            centroids[:, 0] * centroids[:, 0] + centroids[:, 1] * centroids[:, 1]
        )
        concentric = zeros_like(idomain_top)
        concentric.values = np.where(dist < 500, 0, 1)
        concentric["name"] = "concentric"
        return concentric


class WellCases:
    def case_one_well(self):
        return imod.mf6.Well(
            x=[500.0], y=[0.0], screen_top=[3.0], screen_bottom=[2.0], rate=[1.0]
        )

    def case_all_wells(self, idomain_top):
        x = idomain_top.ugrid.grid.face_x
        y = idomain_top.ugrid.grid.face_y
        size = len(x)

        return imod.mf6.Well(
            x=x,
            y=y,
            screen_top=size * [3.0],
            screen_bottom=size * [2.0],
            rate=size * [1.0],
            minimum_k=1e-19,
        )


class HorizontalFlowBarrierCases:
    def case_hfb_vertical(self):
        # Vertical line at x = -100
        barrier_y = [-990.0, 990.0]
        barrier_x = [-100.0, -100.0]

        return gpd.GeoDataFrame(
            geometry=[shapely.linestrings(barrier_x, barrier_y)],
            data={
                "resistance": [10.0],
                "ztop": [10.0],
                "zbottom": [0.0],
            },
        )

    def case_hfb_horizontal(self):
        # Horizontal line at y = -100.0
        barrier_x = [-990.0, 990.0]
        barrier_y = [-100.0, -100.0]

        return gpd.GeoDataFrame(
            geometry=[shapely.linestrings(barrier_x, barrier_y)],
            data={
                "resistance": [10.0],
                "ztop": [10.0],
                "zbottom": [0.0],
            },
        )

    def case_hfb_horizontal_outside_domain(self):
        # Horizontal line at y = -100.0 running outside domain
        barrier_x = [-990.0, 10_000.0]
        barrier_y = [-100.0, -100.0]

        return gpd.GeoDataFrame(
            geometry=[shapely.linestrings(barrier_x, barrier_y)],
            data={
                "resistance": [10.0],
                "ztop": [10.0],
                "zbottom": [0.0],
            },
        )

    def case_hfb_horizontal_origin(self):
        # Horizontal line through origin
        barrier_x = [-990.0, 990.0]
        barrier_y = [0.0, 0.0]

        return gpd.GeoDataFrame(
            geometry=[shapely.linestrings(barrier_x, barrier_y)],
            data={
                "resistance": [10.0],
                "ztop": [10.0],
                "zbottom": [0.0],
            },
        )

    def case_hfb_diagonal(self):
        # Diagonal line
        barrier_y = [-480.0, 480.0]
        barrier_x = [-480.0, 480.0]

        return gpd.GeoDataFrame(
            geometry=[shapely.linestrings(barrier_x, barrier_y)],
            data={
                "resistance": [10.0],
                "ztop": [10.0],
                "zbottom": [0.0],
            },
        )


@pytest.mark.usefixtures("circle_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_partitioning_unstructured(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_array: xu.UgridDataArray
):
    # %%
    simulation = circle_model
    # Increase the recharge to make the head gradient more pronounced.
    simulation["GWF_1"]["rch"]["rate"] *= 100

    # Run the original example, so without partitioning, and save the simulation
    # results.
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)
    simulation.run()

    expected_head = simulation.open_head(
        simulation_start_time=np.datetime64("1999-01-01"),
        time_unit="d",
    )
    expected_flow_cbc = simulation.open_flow_budget(
        simulation_start_time=np.datetime64("1999-01-01"),
        time_unit="d",
    )

    # Partition the simulation, run it, and save the (merged) results.
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()  # %%
    actual_head = split_simulation.open_head(
        simulation_start_time="01-01-1999", time_unit="d"
    )
    actual_head = actual_head.ugrid.reindex_like(expected_head)

    actual_flow_cbc = split_simulation.open_flow_budget(
        simulation_start_time=np.datetime64("1999-01-01"), time_unit="d"
    )
    actual_flow_cbc = actual_flow_cbc.ugrid.reindex_like(expected_flow_cbc)

    # Compare the head result of the original simulation with the result of the partitioned simulation.
    np.testing.assert_allclose(
        actual_head["head"].values, expected_head.values, rtol=1e-5, atol=1e-3
    )
    for key in ["flow-lower-face", "flow-horizontal-face"]:
        if partition_array["name"].values[()] == "concentric" and key in [
            "flow-horizontal-face"
        ]:
            continue
        np.testing.assert_allclose(
            actual_flow_cbc[key].values,
            expected_flow_cbc[key].values,
            rtol=1e-5,
            atol=1e-3,
        )


@pytest.mark.usefixtures("circle_model")
@pytest.mark.parametrize("inactivity_marker", [0, -1])
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_partitioning_unstructured_with_inactive_cells(
    tmp_path: Path,
    circle_model: Modflow6Simulation,
    partition_array: xu.UgridDataArray,
    inactivity_marker: int,
):
    simulation = circle_model

    # Increase the recharge to make the head gradient more pronounced
    simulation["GWF_1"]["rch"]["rate"] *= 100

    # Deactivate some cells on idomain
    idomain = simulation["GWF_1"].domain
    deactivated_cells = slice(93, 101)
    idomain.loc[{"mesh2d_nFaces": deactivated_cells}] = inactivity_marker

    # The cells we just deactivated on idomain must be deactivated on package inputs too.
    simulation["GWF_1"].mask_all_packages(idomain)

    # Run the original example, so without partitioning, and save the simulation
    # results
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)

    simulation.run()

    expected_head = simulation.open_head()

    # TODO: Fix issue 669
    #    original_cbc = imod.mf6.open_cbc(
    #        original_dir / "GWF_1/GWF_1.cbc",
    #        original_dir / "GWF_1/disv.disv.grb",
    #    )

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    actual_head = split_simulation.open_head()
    actual_head = actual_head.ugrid.reindex_like(expected_head)
    # _ = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(
        actual_head["head"].values, expected_head.values, rtol=1e-5, atol=1e-3
    )


@pytest.mark.usefixtures("circle_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
@parametrize_with_cases("hfb", cases=HorizontalFlowBarrierCases)
def test_partitioning_unstructured_hfb(
    tmp_path: Path,
    circle_model: Modflow6Simulation,
    partition_array: xu.UgridDataArray,
    hfb: imod.mf6.HorizontalFlowBarrierBase,
):
    simulation = circle_model
    # Increase the recharge to make the head gradient more pronounced
    simulation["GWF_1"]["rch"]["rate"] *= 100

    simulation["GWF_1"]["hfb"] = imod.mf6.HorizontalFlowBarrierResistance(geometry=hfb)

    # Run the original example, so without partitioning, and save the simulation
    # results
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)
    simulation.run()

    expected_head = simulation.open_head()
    expected_flow_cbc = simulation.open_flow_budget()

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    actual_head = split_simulation.open_head()

    actual_flow_cbc = split_simulation.open_flow_budget()

    actual_head = actual_head.ugrid.reindex_like(expected_head)
    actual_flow_cbc = actual_flow_cbc.ugrid.reindex_like(expected_flow_cbc)

    # Compare the head result of the original simulation with the result of the
    # partitioned simulation. Criteria are a bit looser than in other tests
    # because we are dealing with a problem with heads ranging roughly from 20
    # m to 0 m, and the HFB adds extra complexity to this.
    np.testing.assert_allclose(
        actual_head["head"].values, expected_head.values, rtol=0.005
    )
    for key in ["flow-lower-face", "flow-horizontal-face"]:
        if partition_array["name"].values[()] == "concentric" and key in [
            "flow-horizontal-face"
        ]:
            continue
        np.testing.assert_allclose(
            actual_flow_cbc[key].values,
            expected_flow_cbc[key].values,
            rtol=1.0,
            atol=31.66250038,
        )


@pytest.mark.usefixtures("circle_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
@parametrize_with_cases("well", cases=WellCases)
def test_partitioning_unstructured_with_well(
    tmp_path: Path,
    circle_model: Modflow6Simulation,
    partition_array: xu.UgridDataArray,
    well: imod.mf6.Well,
):
    simulation = circle_model
    # Increase the recharge to make the head gradient more pronounced
    simulation["GWF_1"]["rch"]["rate"] *= 100

    # Add well
    simulation["GWF_1"]["well"] = well

    # Run the original example, so without partitioning, and save the simulation
    # results.
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)
    simulation.run()

    expected_head = simulation.open_head()
    expected_flow_cbc = simulation.open_flow_budget()

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    actual_head = split_simulation.open_head()
    actual_head = actual_head.ugrid.reindex_like(expected_head)
    # TODO:
    # Uncomment when fixed: https://gitlab.com/deltares/imod/imod-python/-/issues/683
    actual_cbc = split_simulation.open_flow_budget()
    actual_cbc = actual_cbc.ugrid.reindex_like(expected_flow_cbc)
    # Compare the head result of the original simulation with the result of the
    # partitioned simulation.
    np.testing.assert_allclose(
        actual_head["head"].values, expected_head.values, rtol=1e-5, atol=1e-3
    )
    for key in ["flow-lower-face", "flow-horizontal-face"]:
        np.testing.assert_allclose(
            actual_cbc[key].values,
            expected_flow_cbc[key].values,
            rtol=1,
            atol=0.01615156,
        )


@pytest.mark.usefixtures("circle_model_transport")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_partition_transport(
    tmp_path: Path,
    circle_model_transport: Modflow6Simulation,
    partition_array: xu.UgridDataArray,
):
    circle_model_transport.write(tmp_path)
    circle_model_transport.run()
    expected_concentration = circle_model_transport.open_concentration()
    expected_budget = circle_model_transport.open_transport_budget(["salinity"])
    expected_budget = expected_budget.sel(time=364)

    new_circle_model = circle_model_transport.split(partition_array)
    new_circle_model.write(tmp_path / "split", binary=False)
    new_circle_model.run()
    actual_concentration = new_circle_model.open_concentration()

    actual_concentration = actual_concentration.ugrid.reindex_like(
        expected_concentration
    )
    np.testing.assert_allclose(
        expected_concentration.values,
        actual_concentration["concentration"].values,
        rtol=7e-5,
        atol=3e-3,
    )    



    # Compare the budgets before and after splitting the simulation.
    # The tolerance is set as the maximum value in the  gwt-gwt
    # exchange as this contains mass transport in the split case that would
    # be in one of the other arrays in the unsplit case.
    actual_budget = new_circle_model.open_transport_budget(["salinity"])
    actual_budget = actual_budget.sel(time=364)    
    actual_budget = actual_budget.ugrid.reindex_like(expected_budget)

    # create a cell-aray of booleans that is true on the exchange boundary cells and false in other locations
    is_exchange_cell = actual_budget["gwt-gwt"].where(actual_budget["gwt-gwt"]!=0).notnull()
    is_exchange_cell = is_exchange_cell.sel(layer=1)

    # create a edge-aray of booleans that is true on the exchange boundary edges and false in other locations    
    face_edge = is_exchange_cell.ugrid.grid.edge_face_connectivity
    face_1 = is_exchange_cell.values[face_edge[:,0]]
    face_2 = is_exchange_cell.values[face_edge[:,1]] 
    is_exchange_edge = zeros_like(expected_budget["flow-horizontal-face"]).sel(layer=1)
    is_exchange_edge.values = np.where(face_1 & face_2, 1,0)
    is_exchange_edge = is_exchange_edge.astype(bool)
          


    for budget_term in (
        "ssm",
        "flow-lower-face",
        "storage-aqueous",
        "flow-horizontal-face",
    ):
        marker = is_exchange_cell
        if budget_term ==  "flow-horizontal-face":
            marker = is_exchange_edge

        np.testing.assert_allclose(
            expected_budget[budget_term].where(~marker,0).values,
            actual_budget[budget_term].where(~marker,0).values,
            rtol=0.3,
            atol=3e-3,
        )


@pytest.mark.usefixtures("circle_model_transport_multispecies_variable_density")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_partition_transport_multispecies(
    tmp_path: Path,
    circle_model_transport_multispecies_variable_density: Modflow6Simulation,
    partition_array: xu.UgridDataArray,
):
    # TODO: put buoyancy package back
    circle_model_transport_multispecies_variable_density["GWF_1"].pop("buoyancy")
    circle_model_transport_multispecies = (
        circle_model_transport_multispecies_variable_density
    )
    circle_model_transport_multispecies.write(tmp_path / "original")
    circle_model_transport_multispecies.run()
    expected_conc = circle_model_transport_multispecies.open_concentration()
    expected_head = circle_model_transport_multispecies.open_head()
    expected_flow_budget = circle_model_transport_multispecies.open_flow_budget(
        simulation_start_time=np.datetime64("1999-01-01"),
        time_unit="d",
    )
    expected_transport_budget = (
        circle_model_transport_multispecies.open_transport_budget(["salt", "temp"])
    )

    # split and run the simulation
    new_circle_model = circle_model_transport_multispecies.split(partition_array)
    new_circle_model.write(tmp_path / "split")
    new_circle_model.run()

    # open results
    actual_conc = new_circle_model.open_concentration()
    actual_head = new_circle_model.open_head()
    flow_budget_new = new_circle_model.open_flow_budget(
        simulation_start_time=np.datetime64("1999-01-01"),
        time_unit="d",
    )
    actual_transport_budget = new_circle_model.open_transport_budget(["salt", "temp"])

    # reindex results
    actual_head = actual_head.ugrid.reindex_like(expected_head)
    actual_conc = actual_conc.ugrid.reindex_like(expected_conc)
    actual_transport_budget = actual_transport_budget.ugrid.reindex_like(
        expected_transport_budget
    )
    actual_flow_budget = flow_budget_new.ugrid.reindex_like(expected_flow_budget)

    # compare simulation results
    np.testing.assert_allclose(
        expected_conc.values, actual_conc["concentration"].values, rtol=4e-4, atol=5e-3
    )
    np.testing.assert_allclose(
        expected_head.values, actual_head["head"].values, rtol=1e-5, atol=1e-3
    )
    # Compare the budgets.
    # The tolerance is set as the maximum value in the  gwf-gwf or gwt-gwt
    # exchange as this contains flow/transport in the split case that would
    # be in one of the other arrays in the unsplit case.
    for key in ["flow-lower-face", "flow-horizontal-face", "sto-ss", "rch"]:
        atol = actual_flow_budget["gwf-gwf"].values.max()[()] * 1.0001
        np.testing.assert_allclose(
            expected_flow_budget[key].values,
            actual_flow_budget[key].values,
            rtol=9,
            atol=atol,
        )
    for key in ["flow-lower-face", "flow-horizontal-face", "storage-aqueous", "ssm"]:
        atol = actual_transport_budget["gwt-gwt"].values.max()[()] * 1.0001
        np.testing.assert_allclose(
            expected_transport_budget[key].values,
            actual_transport_budget[key].values,
            rtol=9,
            atol=atol,
        )


@pytest.mark.usefixtures("circle_model")
def test_slice_model_twice(tmp_path, circle_model):
    flow_model = circle_model["GWF_1"]
    active = flow_model.domain

    submodel_labels = zeros_like(active)
    submodel_labels = submodel_labels.drop_vars("layer")
    submodel_labels.values[:, 50:] = 1
    submodel_labels = submodel_labels.sel(layer=0, drop=True)

    split_simulation_1 = circle_model.split(submodel_labels)
    split_simulation_1.write(
        tmp_path / "split_simulation_1",
        binary=False,
        validate=True,
        use_absolute_paths=False,
    )
    split_simulation_2 = circle_model.split(submodel_labels)
    split_simulation_2.write(
        tmp_path / "split_simulation_2",
        binary=False,
        validate=True,
        use_absolute_paths=False,
    )

    # check that text output was not affected by splitting
    diff = dircmp(tmp_path / "split_simulation_1", tmp_path / "split_simulation_2")
    assert len(diff.diff_files) == 0
    assert len(diff.left_only) == 0
    assert len(diff.right_only) == 0
