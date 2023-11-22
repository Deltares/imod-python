from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_almost_equal

import imod
from imod.mf6 import Modflow6Simulation
from imod.mf6.partitioned_simulation_postprocessing import merge_balances, merge_heads
from imod.mf6.wel import Well
from imod.typing.grid import zeros_like


def setup_partitioning_arrays(idomain_top: xr.DataArray) -> Dict[str, xr.DataArray]:
    result = {}
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
    result["diagonal_1"] = diagonal_submodel_labels_1

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
    result["diagonal_2"] = diagonal_submodel_labels_2

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
    result["four_squares"] = four_squares

    """
    contains a single cell with 3 neighbors in another partitions
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    """

    intrusion = zeros_like(idomain_top)
    intrusion.values[0:15, 0:8] = 0
    intrusion.values[0:15, 8:] = 1
    intrusion.values[8, 8] = 0
    result["intrusion"] = intrusion

    """
    partition forms an island in another partition
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    """
    island = zeros_like(idomain_top)
    island.values[2:5, 2:5] = 1
    result["island"] = island

    return result


@pytest.mark.usefixtures("transient_twri_model")
@pytest.mark.parametrize(
    "partition_name",
    ["diagonal_1", "diagonal_2", "four_squares", "intrusion", "island"],
)
def test_partitioning_structured(
    tmp_path: Path, transient_twri_model: Modflow6Simulation, partition_name: str
):
    simulation = transient_twri_model

    # run the original example, so without partitioning, and save the simulation results
    orig_dir = tmp_path / "original"
    simulation.write(orig_dir, binary=False)
    simulation.run()

    orig_head = imod.mf6.open_hds(
        orig_dir / "GWF_1/GWF_1.hds",
        orig_dir / "GWF_1/dis.dis.grb",
    )

    # partition the simulation, run it, and save the (merged) results
    idomain = simulation["GWF_1"].domain
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    split_simulation = simulation.split(partitioning_arrays[partition_name])

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = merge_heads(tmp_path, split_simulation)
    _ = merge_balances(tmp_path, split_simulation)

    # compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(head.values, orig_head.values, rtol=1e-4, atol=1e-4)


@pytest.mark.usefixtures("transient_twri_model")
@pytest.mark.parametrize(
    "partition_name",
    ["diagonal_1", "diagonal_2", "four_squares", "intrusion", "island"],
)
def test_partitioning_structured_with_inactive_cells(
    tmp_path: Path, transient_twri_model: Modflow6Simulation, partition_name: str
):
    simulation = transient_twri_model
    idomain = simulation["GWF_1"].domain
    idomain.loc[{"x": 32500, "y": slice(67500, 7500)}] = 0
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
                        ] = 0
    # run the original example, so without partitioning, and save the simulation results
    orig_dir = tmp_path / "original"
    simulation.write(orig_dir, binary=False)

    simulation.run()

    orig_head = imod.mf6.open_hds(
        orig_dir / "GWF_1/GWF_1.hds",
        orig_dir / "GWF_1/dis.dis.grb",
    )

    # partition the simulation, run it, and save the (merged) results
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    split_simulation = simulation.split(partitioning_arrays[partition_name])

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = merge_heads(tmp_path, split_simulation)
    _ = merge_balances(tmp_path, split_simulation)

    # compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(head.values, orig_head.values, rtol=1e-4, atol=1e-4)


@pytest.mark.usefixtures("transient_twri_model")
@pytest.mark.parametrize(
    "partition_name",
    ["diagonal_1", "diagonal_2", "four_squares", "intrusion", "island"],
)
def test_partitioning_structured_with_vpt_cells(
    tmp_path: Path, transient_twri_model: Modflow6Simulation, partition_name: str
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

    # run the original example, so without partitioning, and save the simulation results
    orig_dir = tmp_path / "original"
    simulation.write(orig_dir, binary=False)

    simulation.run()

    orig_head = imod.mf6.open_hds(
        orig_dir / "GWF_1/GWF_1.hds",
        orig_dir / "GWF_1/dis.dis.grb",
    )

    # partition the simulation, run it, and save the (merged) results
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    split_simulation = simulation.split(partitioning_arrays[partition_name])

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = merge_heads(tmp_path, split_simulation)
    _ = merge_balances(tmp_path, split_simulation)

    # compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(head.values, orig_head.values, rtol=1e-4, atol=1e-4)


@pytest.mark.usefixtures("transient_twri_model")
def test_partitioning_structured_geometry_auxiliary_variables(
    transient_twri_model: Modflow6Simulation,
):
    simulation = transient_twri_model

    # partition the simulation, run it, and save the (merged) results
    idomain = simulation["GWF_1"].domain
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))
    partition_name = "intrusion"
    split_simulation = simulation.split(partitioning_arrays[partition_name])

    assert_almost_equal(
        split_simulation["split_exchanges"][0].dataset["cdist"].values,
        np.array(
            [
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
            ]
        ),
    )

    assert_almost_equal(
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
@pytest.mark.parametrize(
    "partition_name",
    ["diagonal_1", "diagonal_2", "four_squares", "intrusion", "island"],
)
def test_partitioning_structured_high_level_well(
    tmp_path: Path, transient_twri_model: Modflow6Simulation, partition_name: str
):
    """
    In this test we include a high-level well package with 1 well in it to the
    simulation. The well will be in active in 1 partition and should therefore
    be inactive or non-present in the other partitions This should not give
    validation errors.
    """
    simulation = transient_twri_model

    # Create and fill the groundwater model.
    simulation["GWF_1"]["wel"] = imod.mf6.Well(
        x=[52500.0],
        y=[52500.0],
        screen_top=[-300.0],
        screen_bottom=[-450.0],
        rate=[-5.0],
        minimum_k=1e-19,
    )

    # run the original example, so without partitioning, and save the simulation results
    orig_dir = tmp_path / "original"
    simulation.write(orig_dir, binary=False)
    simulation.run()

    orig_head = imod.mf6.open_hds(
        orig_dir / "GWF_1/GWF_1.hds",
        orig_dir / "GWF_1/dis.dis.grb",
    )

    # partition the simulation, run it, and save the (merged) results
    idomain = simulation["GWF_1"].domain
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    split_simulation = simulation.split(partitioning_arrays[partition_name])

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = merge_heads(tmp_path, split_simulation)
    _ = merge_balances(tmp_path, split_simulation)

    # compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(head.values, orig_head.values, rtol=1e-4, atol=1e-4)
