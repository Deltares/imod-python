from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6 import Modflow6Simulation
from imod.mf6.partitioned_simulation_postprocessing import merge_balances, merge_heads
from imod.mf6.wel import Well
from imod.typing.grid import zeros_like
import flopy
from flopy.mf6.utils import Mf6Splitter
def setup_partitioning_arrays(idomain_top: xr.DataArray) -> Dict[str, xr.DataArray]:
    result = {}
    two_parts = zeros_like(idomain_top)

    two_parts.values[:97] = 0
    two_parts.values[97:] = 1

    result["two_parts"] = two_parts

    three_parts = zeros_like(idomain_top)

    three_parts.values[:37] = 0
    three_parts.values[37:97] = 1
    three_parts.values[97:] = 2
    result["three_parts"] = three_parts

    return result
def setup_partitioning_flopy(idomain_top: xr.DataArray) -> Dict[str, xr.DataArray]:
    result = {}
    two_parts = zeros_like(idomain_top)

    two_parts.values[:97] = 0
    two_parts.values[97:] = 1

    result["two_parts"] = two_parts

    three_parts = zeros_like(idomain_top)

    three_parts.values[:37] = 0
    three_parts.values[37:97] = 1
    three_parts.values[97:] = 2
    result["three_parts"] = three_parts

    return result

@pytest.mark.usefixtures("circle_model")
@pytest.mark.parametrize(
    "partition_name",
    ["two_parts", "three_parts"],
)
def test_partitioning_unstructured_flopy(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_name: str
):
    simulation = circle_model
    # increase the recharge to make the head gradient more pronounced
    simulation["GWF_1"]["rch"]["rate"] *= 100
    idomain = simulation["GWF_1"].domain
    partitioning_arrays = setup_partitioning_flopy(idomain.isel(layer=0))
    partitioning_array = partitioning_arrays["two_parts"]
    # run the original example, so without partitioning, and save the simulation results
    orig_dir = tmp_path / "original"
    simulation.write(orig_dir, binary=False, use_absolute_paths=True)

    split_ip_dir = tmp_path / "split_ip"    
    split_simulation_ip = simulation.split(partitioning_arrays[partition_name])
    split_simulation_ip.write(split_ip_dir, binary=False)

    flopy_sim =flopy.mf6.MFSimulation.load( sim_ws=orig_dir, verbosity_level=1,)
    flopy_dir = tmp_path / "flopy"
    flopy_sim.set_sim_path(flopy_dir)
    flopy_sim.write_simulation(silent=False)
    flopy_sim.run_simulation(silent=True)


    gwf = flopy_sim.get_model()


    mf_splitter = Mf6Splitter(flopy_sim)
   
    
    flopy_split_sim =  mf_splitter.split_model(partitioning_array)
    flopy_split_dir = tmp_path / "flopy_split"   
    flopy_split_sim.set_sim_path(flopy_split_dir)
    flopy_split_sim.write_simulation(silent=False)
    flopy_split_sim.run_simulation(silent=False)
    