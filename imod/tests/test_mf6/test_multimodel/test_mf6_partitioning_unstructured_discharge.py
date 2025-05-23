import uuid
from pathlib import Path

import numpy as np
import pytest
import xugrid as xu
from pytest_cases import parametrize_with_cases

from imod.mf6 import Modflow6Simulation
from imod.typing import UnstructuredData
from imod.typing.grid import zeros_like


def reduce_coordinate_precision(ugrid: UnstructuredData) -> None:
    """
    Reduces the precision of x and y coordinates in an ugrid dataset to 5 decimals.
    """
    ugrid.ugrid.grid.node_x = ugrid.ugrid.grid.node_x.round(5)
    ugrid.ugrid.grid.node_y = ugrid.ugrid.grid.node_y.round(5)


def save_and_load(tmp_path: Path, ugrid: UnstructuredData) -> xu.UgridDataset:
    """
    Saves an ugrid dataset to file, loads the resulting file and returns the ugrid dataset in it.
    """
    filename = tmp_path / str(uuid.uuid4())
    ugrid.ugrid.to_netcdf(filename)
    ugrid = xu.open_dataset(filename)
    return ugrid


@pytest.fixture(scope="function")
def idomain_top(circle_model):
    idomain = circle_model["GWF_1"].domain
    return idomain.isel(layer=0)


class PartitionArrayCases:
    def case_two_parts(self, idomain_top) -> xu.UgridDataArray:
        two_parts = zeros_like(idomain_top)
        two_parts.values[74:] = 1
        return two_parts

    def case_two_parts2(self, idomain_top) -> xu.UgridDataArray:
        two_parts = zeros_like(idomain_top)
        two_parts.values[47:] = 1
        return two_parts

    def case_two_parts3(self, idomain_top) -> xu.UgridDataArray:
        two_parts = zeros_like(idomain_top)
        two_parts.values[147:] = 1
        return two_parts

    def case_two_parts4(self, idomain_top) -> xu.UgridDataArray:
        two_parts = zeros_like(idomain_top)
        two_parts.values[187:] = 1
        return two_parts

    def case_two_parts_inverse(self, idomain_top) -> xu.UgridDataArray:
        two_parts_inverse = zeros_like(idomain_top)
        two_parts_inverse.values[:74] = 1
        return two_parts_inverse

    def case_two_parts_inverse2(self, idomain_top) -> xu.UgridDataArray:
        two_parts_inverse = zeros_like(idomain_top)
        two_parts_inverse.values[:47] = 1
        return two_parts_inverse

    def case_two_parts_inverse3(self, idomain_top) -> xu.UgridDataArray:
        two_parts_inverse = zeros_like(idomain_top)
        two_parts_inverse.values[:147] = 1
        return two_parts_inverse

    def case_two_parts_inverse4(self, idomain_top) -> xu.UgridDataArray:
        two_parts_inverse = zeros_like(idomain_top)
        two_parts_inverse.values[:187] = 1
        return two_parts_inverse

    def case_three_parts(self, idomain_top) -> xu.UgridDataArray:
        three_parts = zeros_like(idomain_top)
        three_parts.values[72:144] = 1
        three_parts.values[144:] = 2
        return three_parts


@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_specific_discharge_results(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_array: xu.UgridDataArray
):
    simulation = circle_model
    label_array = partition_array

    # Turn on specific discharge calculation
    simulation["GWF_1"]["npf"].dataset["save_specific_discharge"] = True

    # Write the original simulation (unsplit) to file.
    ip_unsplit_dir = tmp_path / "original"
    simulation.write(ip_unsplit_dir, binary=False, use_absolute_paths=True)

    # Split the original simulation and write it to file
    ip_dir = tmp_path / "ip_split"
    new_sim = simulation.split(label_array)
    new_sim.write(ip_dir, False)

    # Run the simulations and load the balance results
    simulation.run()
    new_sim.run()
    original_balances = simulation.open_flow_budget()
    original_heads = simulation.open_head()
    split_balances = new_sim.open_flow_budget()
    split_head = new_sim.open_head()

    # Reduce the coordinate precision of the heads and balances
    reduce_coordinate_precision(split_head)
    reduce_coordinate_precision(original_heads)
    reduce_coordinate_precision(split_balances["npf-qx"])
    reduce_coordinate_precision(original_balances["npf-qx"])
    reduce_coordinate_precision(split_balances["npf-qy"])
    reduce_coordinate_precision(original_balances["npf-qy"])

    # Reload the ugrid dataarrays to avoid issues with the subsequent reindexing
    split_balances_x_v2 = save_and_load(tmp_path, split_balances["npf-qx"])
    split_balances_y_v2 = save_and_load(tmp_path, split_balances["npf-qy"])
    original_balances_x_v2 = save_and_load(tmp_path, original_balances["npf-qx"])
    original_balances_y_v2 = save_and_load(tmp_path, original_balances["npf-qy"])

    # Reindex the arrays that come from modflow output to match the indexing of unsplit results
    split_head = split_head.ugrid.reindex_like(original_heads)
    split_balances_x_v2 = split_balances_x_v2.ugrid.reindex_like(original_balances_x_v2)
    split_balances_y_v2 = split_balances_y_v2.ugrid.reindex_like(original_balances_y_v2)

    # Compute differences in head and specific discharge results
    head_diff = original_heads.isel(layer=0, time=-1) - split_head.isel(
        layer=0, time=-1
    )
    veldif_x = original_balances_x_v2["npf-qx"] - split_balances_x_v2["npf-qx"]
    veldif_y = original_balances_y_v2["npf-qy"] - split_balances_y_v2["npf-qy"]

    # Assert results are close for head and specific discharge
    assert np.abs(head_diff["head"].values).max() < 1e-5
    assert np.abs(veldif_x.values).max() < 1e-6
    assert np.abs(veldif_y.values).max() < 1e-6
