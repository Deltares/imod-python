from __future__ import annotations
from filecmp import dircmp

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr
from pytest_cases import case, parametrize_with_cases

import imod
from imod.mf6 import Modflow6Simulation
from imod.mf6.wel import Well
from imod.typing.grid import zeros_like


@pytest.mark.usefixtures("transient_twri_model")
@pytest.fixture(scope="function")
def idomain_top(transient_twri_model):
    idomain = transient_twri_model["GWF_1"].domain
    return idomain.isel(layer=0)


class PartitionArrayCases:
    def case_diagonal_array_1(self, idomain_top):
        diagonal_submodel_labels_1 = zeros_like(idomain_top)

        """
        a diagonal partition boundary
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        """
        for i in range(15):
            for j in range(i):
                diagonal_submodel_labels_1.values[i, j] = 1
        return diagonal_submodel_labels_1

    def case_diagonal_array_2(self, idomain_top):
        """
        another diagonal boundary
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        """
        diagonal_submodel_labels_2 = zeros_like(idomain_top)
        for i in range(15):
            for j in range(15 - i):
                diagonal_submodel_labels_2.values[i, j] = 1
        return diagonal_submodel_labels_2

    def case_four_squares(self, idomain_top):
        """
        4 square domains
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
        """
        four_squares = zeros_like(idomain_top)
        four_squares.values[0:7, 0:7] = 0
        four_squares.values[0:7, 7:] = 1
        four_squares.values[7:, 0:7] = 2
        four_squares.values[7:, 7:] = 3
        return four_squares

    @case(tags="intrusion")
    def case_intrusion(self, idomain_top):
        """
        Contains a single cell with 3 neighbors in another partitions.
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        """

        intrusion = zeros_like(idomain_top)
        intrusion.values[0:15, 0:8] = 0
        intrusion.values[0:15, 8:] = 1
        intrusion.values[8, 8] = 0
        return intrusion

    def case_island(self, idomain_top):
        """
        Partition forms an island in another partition
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        """
        island = zeros_like(idomain_top)
        island.values[2:5, 2:5] = 1
        return island


class WellCases:
    def case_one_well(self):
        return imod.mf6.Well(
            x=[52500.0],
            y=[52500.0],
            screen_top=[-300.0],
            screen_bottom=[-450.0],
            rate=[-5.0],
            minimum_k=1e-19,
        )

    def case_all_well(self, idomain_top):
        x = idomain_top.coords["x"].values
        y = idomain_top.coords["y"].values
        x_mesh, y_mesh = np.meshgrid(x, y)
        x_list = x_mesh.ravel()
        y_list = y_mesh.ravel()
        size = len(x_list)

        return imod.mf6.Well(
            x=x_list,
            y=y_list,
            screen_top=size * [-300.0],
            screen_bottom=size * [-450.0],
            rate=size * [-5.0],
            minimum_k=1e-19,
        )


class HorizontalFlowBarrierCases:
    def case_hfb_vertical(self):
        # Vertical line at x = 52500.0
        barrier_y = [0.0, 52500.0]
        barrier_x = [52500.0, 52500.0]

        return gpd.GeoDataFrame(
            geometry=[shapely.linestrings(barrier_x, barrier_y)],
            data={
                "resistance": [10.0],
                "ztop": [10.0],
                "zbottom": [0.0],
            },
        )

    def case_hfb_horizontal(self):
        # Horizontal line at y = 52500.0
        barrier_x = [0.0, 52500.0]
        barrier_y = [52500.0, 52500.0]

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
        barrier_x = [0.0, 1_000_000.0]
        barrier_y = [52500.0, 52500.0]

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
        barrier_y = [0.0, 52500.0]
        barrier_x = [0.0, 52500.0]

        return gpd.GeoDataFrame(
            geometry=[shapely.linestrings(barrier_x, barrier_y)],
            data={
                "resistance": [10.0],
                "ztop": [10.0],
                "zbottom": [0.0],
            },
        )


@pytest.mark.usefixtures("transient_twri_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_partitioning_structured(
    tmp_path: Path,
    transient_twri_model: Modflow6Simulation,
    partition_array: xr.DataArray,
):
    simulation = transient_twri_model

    # Run the original example, so without partitioning, and save the simulation
    # results.
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)
    simulation.run()

    original_head = imod.mf6.open_hds(
        original_dir / "GWF_1/GWF_1.hds",
        original_dir / "GWF_1/dis.dis.grb",
    )

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    _ = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the
    # partitioned simulation.
    np.testing.assert_allclose(
        head["head"].values, original_head.values, rtol=1e-4, atol=1e-4
    )

@pytest.mark.usefixtures("transient_twri_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases, glob="*four_squares")
def test_split_dump(
    tmp_path: Path,
    transient_twri_model: Modflow6Simulation,
    partition_array: xr.DataArray,
):
    simulation = transient_twri_model
    split_simulation = simulation.split(partition_array)
    split_simulation.dump(tmp_path/"split")
    reloaded_split = imod.mf6.Modflow6Simulation.from_file(tmp_path/"split/ex01-twri_partioned.toml")

    split_simulation.write(tmp_path/"split/original")
    reloaded_split.write(tmp_path/"split/reloaded")

    diff = dircmp(tmp_path/"split/original", tmp_path/"split/reloaded")
    assert len(diff.diff_files) == 0
    assert len(diff.left_only) == 0
    assert len(diff.right_only) == 0    

@pytest.mark.usefixtures("transient_twri_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases.case_four_squares)
def test_partitioning_write_after_split(
    tmp_path: Path,
    transient_twri_model: Modflow6Simulation,
    partition_array: xr.DataArray,
):
    simulation = transient_twri_model

    # write simulation before splitting, then split simulation, then write the simulation a second time
    simulation.write(tmp_path/ "first_time", binary=False)

    _ = simulation.split(partition_array)

    simulation.write(tmp_path/ "second_time", binary=False)

    #check that text output was not affected by splitting
    diff = dircmp(tmp_path/"first_time", tmp_path/"second_time")
    assert len(diff.diff_files) == 0
    assert len(diff.left_only) == 0
    assert len(diff.right_only) == 0    
    
@pytest.mark.usefixtures("transient_twri_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_partitioning_structured_with_inactive_cells(
    tmp_path: Path,
    transient_twri_model: Modflow6Simulation,
    partition_array: xr.DataArray,
):
    simulation = transient_twri_model
    idomain = simulation["GWF_1"].domain
    idomain.loc[{"x": 32500, "y": slice(67500, 7500)}] = 0
    for _, package in simulation["GWF_1"].items():
        if not isinstance(package, Well):
            for arrayname in package.dataset.keys():
                if "x" in package[arrayname].coords:
                    if np.issubdtype(package[arrayname].dtype, float):
                        package[arrayname].loc[
                            {"x": 32500, "y": slice(67500, 7500)}
                        ] = np.nan
                    else:
                        package[arrayname].loc[
                            {"x": 32500, "y": slice(67500, 7500)}
                        ] = 0
    # Run the original example, so without partitioning, and save the simulation
    # results.
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)

    simulation.run()

    original_head = imod.mf6.open_hds(
        original_dir / "GWF_1/GWF_1.hds",
        original_dir / "GWF_1/dis.dis.grb",
    )

    # Partition the simulation, run it, and save the (merged) results.
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    _ = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the
    # partitioned simulation.
    np.testing.assert_allclose(
        head["head"].values, original_head.values, rtol=1e-4, atol=1e-4
    )


@pytest.mark.usefixtures("transient_twri_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_partitioning_structured_with_vpt_cells(
    tmp_path: Path,
    transient_twri_model: Modflow6Simulation,
    partition_array: xr.DataArray,
):
    simulation = transient_twri_model
    idomain = simulation["GWF_1"].domain
    idomain.loc[{"x": 32500, "y": slice(67500, 7500)}] = -1

    for name, package in simulation["GWF_1"].items():
        if not isinstance(package, Well):
            for arrayname in package.dataset.keys():
                if "x" in package[arrayname].coords:
                    if np.issubdtype(package[arrayname].dtype, float):
                        package[arrayname].loc[
                            {"x": 32500, "y": slice(67500, 7500)}
                        ] = np.nan
                    else:
                        package[arrayname].loc[
                            {"x": 32500, "y": slice(67500, 7500)}
                        ] = -1

    # Run the original example, so without partitioning, and save the simulation
    # results.
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)

    simulation.run()

    original_head = imod.mf6.open_hds(
        original_dir / "GWF_1/GWF_1.hds",
        original_dir / "GWF_1/dis.dis.grb",
    )

    # Partition the simulation, run it, and save the (merged) results.
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    _ = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the
    # partitioned simulation.
    np.testing.assert_allclose(
        head["head"].values, original_head.values, rtol=1e-4, atol=1e-4
    )


@pytest.mark.usefixtures("transient_twri_model")
@parametrize_with_cases(
    "partition_array", cases=PartitionArrayCases, has_tag="intrusion"
)
def test_partitioning_structured_geometry_auxiliary_variables(
    transient_twri_model: Modflow6Simulation, partition_array: xr.DataArray
):
    simulation = transient_twri_model

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    np.testing.assert_almost_equal(
        split_simulation["split_exchanges"][0].dataset["cdist"].values, 5000.0
    )
    np.testing.assert_almost_equal(
        split_simulation["split_exchanges"][0].dataset["angldegx"],
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                90.0,
                0.0,
                270.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                90.0,
                0.0,
                270.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                90.0,
                0.0,
                270.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )


@pytest.mark.usefixtures("transient_twri_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
@parametrize_with_cases("well", cases=WellCases)
def test_partitioning_structured_one_high_level_well(
    tmp_path: Path,
    transient_twri_model: Modflow6Simulation,
    partition_array: xr.DataArray,
    well: imod.mf6.Well,
):
    """
    In this test we include a high-level well package with 1 well in it to the
    simulation. The well will be in active in 1 partition and should therefore
    be inactive or non-present in the other partitions This should not give
    validation errors.
    """
    simulation = transient_twri_model

    # Create and fill the groundwater model.
    simulation["GWF_1"]["wel"] = well

    # Run the original example, so without partitioning, and save the simulation
    # results.
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)
    simulation.run()

    original_head = imod.mf6.open_hds(
        original_dir / "GWF_1/GWF_1.hds",
        original_dir / "GWF_1/dis.dis.grb",
    )

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    _ = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the
    # partitioned simulation.
    np.testing.assert_allclose(
        head["head"].values, original_head.values, rtol=1e-4, atol=1e-4
    )


def is_expected_hfb_partition_combination_fail(current_cases):
    """
    Helper function for all expected failures

    Idea taken from:
    https://github.com/smarie/python-pytest-cases/issues/195#issuecomment-834232905
    """
    # In this combination the hfb lays along partition model domain.
    if (current_cases["partition_array"].id == "four_squares") and (
        current_cases["hfb"].id == "hfb_diagonal"
    ):
        return True
    return False


@pytest.mark.usefixtures("transient_twri_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
@parametrize_with_cases("hfb", cases=HorizontalFlowBarrierCases)
def test_partitioning_structured_hfb(
    tmp_path: Path,
    transient_twri_model: Modflow6Simulation,
    partition_array: xr.DataArray,
    hfb: imod.mf6.HorizontalFlowBarrierResistance,
    current_cases,
):
    """
    In this test we include a high-level hfb package with 1 horizontal flow barrier
    in it to the simulation. The hfb will be in active in 1 partition and should
    therefore be inactive or non-present in the other partitions This should not
    give validation errors.
    """

    simulation = transient_twri_model

    simulation["GWF_1"]["hfb"] = imod.mf6.HorizontalFlowBarrierResistance(geometry=hfb)

    # Run the original example, so without partitioning, and save the simulation
    # results.
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)
    simulation.run()

    original_head = imod.mf6.open_hds(
        original_dir / "GWF_1/GWF_1.hds",
        original_dir / "GWF_1/dis.dis.grb",
    )

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    # Certain combinations are expected to fail
    if is_expected_hfb_partition_combination_fail(current_cases):
        pytest.xfail("Combination hfb - partition_array expected to fail.")

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    _ = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the
    # partitioned simulation.
    np.testing.assert_allclose(head["head"].values, original_head.values, rtol=1e-3)
