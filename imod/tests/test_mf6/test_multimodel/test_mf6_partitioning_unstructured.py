from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6 import Modflow6Simulation
from imod.mf6.wel import Well
from imod.typing.grid import zeros_like
from copy import deepcopy


def setup_partitioning_arrays(idomain_top: xr.DataArray) -> Dict[str, xr.DataArray]:
    result = {}

    end_range = 75
    two_parts = zeros_like(idomain_top)
    two_parts.values[end_range:] = 1

    result["two_parts"] = two_parts

    two_parts_inverse = zeros_like(idomain_top)
    two_parts_inverse.values[:end_range] = 1

    result["two_parts_inverse"] = two_parts_inverse

    three_parts = zeros_like(idomain_top)

    three_parts.values[:72] = 0
    three_parts.values[72:144] = 1
    three_parts.values[144:] = 2
    result["three_parts"] = three_parts

    return result


def setup_reference_results() -> Dict[str, np.ndarray]:
    result = {}
    result["two_parts"] = np.array(
        [
            360.0,
            360.0,
            240.0,
            353.79397689,
            360.0,
            300.0,
            360.0,
            360.0,
            360.0,
            360.0,
            225.0,
            349.63376582,
            219.55330268,
            300.0,
            300.0,
            360.0,
            360.0,
            240.0,
            353.79397689,
            360.0,
            300.0,
            360.0,
            360.0,
            360.0,
            360.0,
            225.0,
            349.63376582,
            219.55330268,
            300.0,
            300.0,
        ]
    )

    result["two_parts_inverse"] = np.array(
        [
            180.0,
            180.0,
            60.0,
            180.0,
            120.0,
            180.0,
            180.0,
            180.0,
            180.0,
            173.79397689,
            45.0,
            169.63376582,
            39.55330268,
            120.0,
            120.0,
            180.0,
            180.0,
            60.0,
            180.0,
            120.0,
            180.0,
            180.0,
            180.0,
            180.0,
            173.79397689,
            45.0,
            169.63376582,
            39.55330268,
            120.0,
            120.0,
        ]
    )
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

    orig_cbc = imod.mf6.open_cbc(
        orig_dir / "GWF_1/GWF_1.cbc",
        orig_dir / "GWF_1/disv.disv.grb",
    )

    # partition the simulation, run it, and save the (merged) results
    idomain = simulation["GWF_1"].domain
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    split_simulation = simulation.split(partitioning_arrays[partition_name])

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    cbc = split_simulation.open_flow_budget()

    # compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(
        head["head"].values, orig_head.values, rtol=1e-5, atol=1e-3
    )
    np.testing.assert_allclose(
        cbc["chd"].values, orig_cbc["chd"].values, rtol=1e-5, atol=1e-3
    )


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

    # TODO: Fix issue 669
    #    orig_cbc = imod.mf6.open_cbc(
    #        orig_dir / "GWF_1/GWF_1.cbc",
    #        orig_dir / "GWF_1/disv.disv.grb",
    #    )

    # partition the simulation, run it, and save the (merged) results
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    split_simulation = simulation.split(partitioning_arrays[partition_name])

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    # _ = split_simulation.open_flow_budget()

    # compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(
        head["head"].values, orig_head.values, rtol=1e-5, atol=1e-3
    )


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

    # TODO: Fix issue 669
    #    orig_cbc = imod.mf6.open_cbc(
    #        orig_dir / "GWF_1/GWF_1.cbc",
    #        orig_dir / "GWF_1/disv.disv.grb",
    #    )

    # partition the simulation, run it, and save the (merged) results
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    split_simulation = simulation.split(partitioning_arrays[partition_name])

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    # _ = split_simulation.open_flow_budget()

    # compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(
        head["head"].values, orig_head.values, rtol=1e-5, atol=1e-3
    )


@pytest.mark.usefixtures("circle_model")
@pytest.mark.parametrize(
    "partition_name",
    ["two_parts", "two_parts_inverse"],
)
def test_partitioning_unstructured_geometric_constants(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_name: str
):
    simulation = circle_model
    simulation.write(tmp_path / "orig", binary=False)
    # increase the recharge to make the head gradient more pronounced
    simulation["GWF_1"]["rch"]["rate"] *= 100
    idomain = simulation["GWF_1"].domain
    partitioning_arrays = setup_partitioning_arrays(idomain.isel(layer=0))

    # partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partitioning_arrays[partition_name])
    split_simulation.write(tmp_path / partition_name, binary=False)
    references = setup_reference_results()

    assert np.allclose(
        split_simulation["split_exchanges"][0]["angldegx"].values,
        references[partition_name],
    )


@pytest.mark.usefixtures("circle_model")
@pytest.mark.parametrize(
    "partition_name",
    ["two_parts", "two_parts_inverse"],
)
def test_specific_discharge_results(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_name: str
):
    simulation = circle_model
    label_arrays = setup_partitioning_arrays(simulation["GWF_1"].domain.isel(layer=0))
    label_array = label_arrays[partition_name]

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

    head_diff = original_heads.isel(layer=0, time=-1) - split_head.isel(layer = 0, time =-1)
    print(f"orig head")
    print( original_heads.isel(layer=0).values)    
    print(f"split head ")
    print(split_head["head"].isel(layer=0).values)
    print(f"head_diff")
    print( head_diff["head"].values)    



    veldif_x = original_balances["npf-qx"] - split_balances["npf-qx"]
    veldif_y = original_balances["npf-qy"] - split_balances["npf-qy"]

    print(f"orig balance x:")
    print( original_balances["npf-qx"].isel(layer=0).values)    
    print(f"split_balances x: ")
    print(split_balances["npf-qx"].isel(layer=0).values)

    print(f"veldif x:")
    print("----------")
    print(veldif_x.isel(layer=0).values)
    print(f"veldif y: {veldif_y.values.max() }")    

    assert veldif_x.values.max() < 1e-6
    assert veldif_y.values.max() < 1e-6

    reldif_x = abs(veldif_x / original_balances["npf-qx"])
    reldif_y = abs(veldif_y / original_balances["npf-qy"])

    print(f"reldif x: {reldif_x.values.max() }")
    print(f"reldif y: {reldif_y.values.max() }")
    assert reldif_x.values.max() < 1e-5
    assert reldif_y.values.max() < 0.12


    pass
