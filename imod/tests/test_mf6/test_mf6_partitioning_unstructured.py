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


@pytest.mark.usefixtures("circle_model")
@pytest.mark.parametrize(
    "partition_name",
    ["two_parts", "three_parts"],
)
def test_partitioning_unstructured(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_name: str
):
    simulation = circle_model
    # increase the recharge to make the head gradient more pronounced
    simulation["GWF_1"]["rch"]["rate"] *= 100

    # run the original example, so without partitioning, and save the simulation results
    orig_dir = tmp_path / "original"
    simulation.write(orig_dir, binary=False)
    simulation.run()

    orig_head = imod.mf6.open_hds(
        orig_dir / "GWF_1/GWF_1.hds",
        orig_dir / "GWF_1/disv.disv.grb",
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
    np.testing.assert_allclose(head.values, orig_head.values, rtol=9e-2, atol=1e-4)


@pytest.mark.usefixtures("circle_model")
@pytest.mark.parametrize(
    "partition_name",
    ["two_parts", "three_parts"],
)
def test_partitioning_unstructured_with_inactive_cells(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_name: str
):
    simulation = circle_model

    # increase the recharge to make the head gradient more pronounced
    simulation["GWF_1"]["rch"]["rate"] *= 100

    # deactivate some cells on idomain
    idomain = simulation["GWF_1"].domain
    deactivated_cells = slice(93, 97)
    idomain.loc[{"mesh2d_nFaces": deactivated_cells}] = 0

    # The cells we just deactivated on idomain must be deactivated on package inputs too.
    for name, package in simulation["GWF_1"].items():
        if not isinstance(package, Well):
            for arrayname in package.dataset.keys():
                if "mesh2d_nFaces" in package[arrayname].coords:
                    if np.issubdtype(package[arrayname].dtype, float):
                        mask_value = np.nan
                    else:
                        mask_value = 0
                    package[arrayname].loc[
                        {"mesh2d_nFaces": deactivated_cells}
                    ] = mask_value

    # run the original example, so without partitioning, and save the simulation results
    orig_dir = tmp_path / "original"
    simulation.write(orig_dir, binary=False)

    simulation.run()

    orig_head = imod.mf6.open_hds(
        orig_dir / "GWF_1/GWF_1.hds",
        orig_dir / "GWF_1/disv.disv.grb",
    )

    # partition the simulation, run it, and save the (merged) results
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    split_simulation = simulation.split(partitioning_arrays[partition_name])

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = merge_heads(tmp_path, split_simulation)

    # compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(head.values, orig_head.values, rtol=1e-1, atol=1e-2)


@pytest.mark.usefixtures("circle_model")
@pytest.mark.parametrize(
    "partition_name",
    ["two_parts", "three_parts"],
)
def test_partitioning_unstructured_with_vpt_cells(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_name: str
):
    simulation = circle_model

    # increase the recharge to make the head gradient more pronounced
    simulation["GWF_1"]["rch"]["rate"] *= 100

    # deactivate some cells on idomain
    idomain = simulation["GWF_1"].domain
    deactivated_cells = slice(93, 101)
    idomain.loc[{"mesh2d_nFaces": deactivated_cells}] = 0

    # The cells we just deactivated on idomain must be deactivated on package inputs too.
    for name, package in simulation["GWF_1"].items():
        if not isinstance(package, Well):
            for arrayname in package.dataset.keys():
                if "mesh2d_nFaces" in package[arrayname].coords:
                    if np.issubdtype(package[arrayname].dtype, float):
                        mask_value = np.nan
                    else:
                        mask_value = 0
                    package[arrayname].loc[
                        {"mesh2d_nFaces": deactivated_cells}
                    ] = mask_value

    # run the original example, so without partitioning, and save the simulation results
    orig_dir = tmp_path / "original"
    simulation.write(orig_dir, binary=False)

    simulation.run()

    orig_head = imod.mf6.open_hds(
        orig_dir / "GWF_1/GWF_1.hds",
        orig_dir / "GWF_1/disv.disv.grb",
    )

    # partition the simulation, run it, and save the (merged) results
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    split_simulation = simulation.split(partitioning_arrays[partition_name])

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = merge_heads(tmp_path, split_simulation)

    # compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(head.values, orig_head.values, rtol=1e-1, atol=1e-2)
