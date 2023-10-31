import copy

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
from imod.mf6.partitioned_simulation_postprocessing import merge_balances, merge_heads
from imod.typing.grid import zeros_like


def setup_partitioning_arrays(idomain_top):
    diagonal_submodel_labels = zeros_like(idomain_top)
    for i in range(15):
        for j in range(i):
            diagonal_submodel_labels.values[i, j] = 1

    four_squares = zeros_like(idomain_top)
    four_squares.values[0:7, 0:7] = 0
    four_squares.values[0:7, 7:] = 1
    four_squares.values[7:, 0:7] = 2
    four_squares.values[7:, 7:] = 3

    intrusion = zeros_like(idomain_top)
    intrusion.values[0:15, 0:7] = 0
    intrusion.values[0:15, 7:] = 1
    intrusion.values[7, 7] = 0

    island = zeros_like(idomain_top)
    island.values[4:7, 4:7] = 1
    return [diagonal_submodel_labels, four_squares, intrusion, island]


@pytest.mark.usefixtures("transient_twri_model")
@pytest.mark.parametrize("partition_index", [0, 1, 2, 3])
def test_partitioning_structured(tmp_path, transient_twri_model, partition_index):
    simulation = transient_twri_model

    # TODO: convert the wells in this fixture to high-level wells
    simulation["GWF_1"].pop("wel")

    # run the original example, so without partitioning, and save the simulation results
    orig_dir = tmp_path / "original"
    simulation.write(orig_dir, binary=False)
    simulation.run()

    orig_head = imod.mf6.open_hds(
        orig_dir / "GWF_1/GWF_1.hds",
        orig_dir / "GWF_1/dis.dis.grb",
    )

    orig_balances = imod.mf6.open_cbc(
        orig_dir / "GWF_1/GWF_1.cbc", orig_dir / "GWF_1/dis.dis.grb"
    )

    # partition the simulation, run it, and save the (merged) results
    idomain = simulation["GWF_1"].domain
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    split_simulation = simulation.split(partitioning_arrays[partition_index])

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = merge_heads(tmp_path, split_simulation)
    balances = merge_balances(tmp_path, split_simulation)

    # compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(head.values, orig_head.values, rtol=1e-4, atol=1e-4)
