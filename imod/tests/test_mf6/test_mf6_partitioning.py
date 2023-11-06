from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6 import Modflow6Simulation
from imod.mf6.partitioned_simulation_postprocessing import merge_balances, merge_heads
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
    intrusion.values[0:15, 0:7] = 0
    intrusion.values[0:15, 7:] = 1
    intrusion.values[7, 7] = 0
    result["intrusion"] = intrusion

    """
    partition forms an island in another partition
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    """
    island = zeros_like(idomain_top)
    island.values[4:7, 4:7] = 1
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
