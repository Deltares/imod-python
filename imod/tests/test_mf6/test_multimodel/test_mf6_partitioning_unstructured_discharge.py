from copy import deepcopy
from pathlib import Path
from typing import Dict

import geopandas as gpd
import imod
import numpy as np
import pytest
import shapely
import xugrid as xu
from imod.mf6 import Modflow6Simulation
from imod.mf6.wel import Well
from imod.typing.grid import zeros_like
from pytest_cases import parametrize_with_cases
import xarray as xr
import uuid
import copy


def reduce_coordinate_precision(ugrid):
    ugrid.ugrid.grid.node_x = ugrid.ugrid.grid.node_x.round(5)
    ugrid.ugrid.grid.node_y = ugrid.ugrid.grid.node_y.round(5)


def save_and_load(tmp_path, ugrid):
    filename = tmp_path / str(uuid.uuid4())
    ugrid.ugrid.to_netcdf(filename)
    ugrid = xu.open_dataset(filename)
    return ugrid


@pytest.mark.usefixtures("circle_model")
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


@pytest.mark.usefixtures("circle_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_specific_discharge_results(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_array: xu.UgridDataArray
):
    simulation = circle_model
    label_array = partition_array

    simulation["GWF_1"]["npf"].dataset["save_specific_discharge"] = True
    ip_unsplit_dir = tmp_path / "original"

    simulation.write(ip_unsplit_dir, binary=False, use_absolute_paths=True)

    # now do the same for imod-python
    ip_dir = tmp_path / "ip_split"
    new_sim = simulation.split(label_array)
    new_sim.write(ip_dir, False)

    simulation.run()
    original_balances = simulation.open_flow_budget()
    original_heads = simulation.open_head()

    new_sim.run()
    split_balances = new_sim.open_flow_budget()
    split_head = new_sim.open_head()
    reduce_coordinate_precision(split_head)
    reduce_coordinate_precision(original_heads)

    split_head = split_head.ugrid.reindex_like(original_heads)

    reduce_coordinate_precision(split_balances["npf-qx"])
    reduce_coordinate_precision(original_balances["npf-qx"])
    reduce_coordinate_precision(split_balances["npf-qy"])
    reduce_coordinate_precision(original_balances["npf-qy"])

    split_balances_x_v2 = save_and_load(tmp_path, split_balances["npf-qx"])
    split_balances_y_v2 = save_and_load(tmp_path, split_balances["npf-qy"])
    original_balances_x_v2 = save_and_load(tmp_path, original_balances["npf-qx"])
    original_balances_y_v2 = save_and_load(tmp_path, original_balances["npf-qy"])

    split_balances_x_v2["npf-qx"] = split_balances_x_v2["npf-qx"].ugrid.reindex_like(
        original_balances_x_v2
    )
    split_balances_y_v2["npf-qy"] = split_balances_y_v2["npf-qy"].ugrid.reindex_like(
        original_balances_y_v2
    )
    head_diff = original_heads.isel(layer=0, time=-1) - split_head.isel(
        layer=0, time=-1
    )

    veldif_x = original_balances_x_v2["data-spdis"] - split_balances_x_v2["npf-qx"]
    veldif_y = original_balances_y_v2["data-spdis"] - split_balances_y_v2["npf-qy"]

    print(f"x: {veldif_x.values.max() }, y: {veldif_y.values.max()}")
    assert veldif_x.values.max() < 1e-6
    assert veldif_y.values.max() < 1e-6

    pass
