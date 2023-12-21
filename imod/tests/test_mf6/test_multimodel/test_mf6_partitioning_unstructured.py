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

@pytest.mark.usefixtures("circle_model")
@pytest.fixture(scope="function")
def idomain_top(circle_model):
    idomain = circle_model["GWF_1"].domain
    return idomain.isel(layer=0)


class PartitionArrayCases:
    def case_two_parts(self, idomain_top) -> xu.UgridDataArray:
        two_parts = zeros_like(idomain_top)
        two_parts.values[108:] = 1
        return two_parts

    def case_two_parts_inverse(self, idomain_top) -> xu.UgridDataArray:
        two_parts_inverse = zeros_like(idomain_top)
        two_parts_inverse.values[:108] = 1
        return two_parts_inverse

    def case_three_parts(self, idomain_top) -> xu.UgridDataArray:
        three_parts = zeros_like(idomain_top)
        three_parts.values[72:144] = 1
        three_parts.values[144:] = 2
        return three_parts


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
    simulation = circle_model
    # Increase the recharge to make the head gradient more pronounced.
    simulation["GWF_1"]["rch"]["rate"] *= 100

    # Run the original example, so without partitioning, and save the simulation
    # results.
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)
    simulation.run()

    original_head = imod.mf6.open_hds(
        original_dir / "GWF_1/GWF_1.hds",
        original_dir / "GWF_1/disv.disv.grb",
    )

    original_cbc = imod.mf6.open_cbc(
        original_dir / "GWF_1/GWF_1.cbc",
        original_dir / "GWF_1/disv.disv.grb",
    )

    # Partition the simulation, run it, and save the (merged) results.
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    cbc = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the partitioned simulation.
    np.testing.assert_allclose(
        head["head"].values, original_head.values, rtol=1e-5, atol=1e-3
    )
    np.testing.assert_allclose(
        cbc["chd"].values, original_cbc["chd"].values, rtol=1e-5, atol=1e-3
    )


@pytest.mark.usefixtures("circle_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_partitioning_unstructured_with_inactive_cells(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_array: xu.UgridDataArray
):
    simulation = circle_model

    # Increase the recharge to make the head gradient more pronounced
    simulation["GWF_1"]["rch"]["rate"] *= 100

    # Deactivate some cells on idomain
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

    # Run the original example, so without partitioning, and save the simulation results
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)

    simulation.run()

    original_head = imod.mf6.open_hds(
        original_dir / "GWF_1/GWF_1.hds",
        original_dir / "GWF_1/disv.disv.grb",
    )

    # TODO: Fix issue 669
    #    original_cbc = imod.mf6.open_cbc(
    #        original_dir / "GWF_1/GWF_1.cbc",
    #        original_dir / "GWF_1/disv.disv.grb",
    #    )

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    # _ = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(
        head["head"].values, original_head.values, rtol=1e-5, atol=1e-3
    )


@pytest.mark.usefixtures("circle_model")
@parametrize_with_cases("partition_array", cases=PartitionArrayCases)
def test_partitioning_unstructured_with_vpt_cells(
    tmp_path: Path, circle_model: Modflow6Simulation, partition_array: xu.UgridDataArray
):
    simulation = circle_model

    # Increase the recharge to make the head gradient more pronounced
    simulation["GWF_1"]["rch"]["rate"] *= 100

    # Deactivate some cells on idomain
    idomain = simulation["GWF_1"].domain
    deactivated_cells = slice(93, 101)
    idomain.loc[{"mesh2d_nFaces": deactivated_cells}] = 0

    # The cells we just deactivated on idomain must be deactivated on package inputs too.
    for _, package in simulation["GWF_1"].items():
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

    # Run the original example, so without partitioning, and save the simulation
    # results
    original_dir = tmp_path / "original"
    simulation.write(original_dir, binary=False)

    simulation.run()

    original_head = imod.mf6.open_hds(
        original_dir / "GWF_1/GWF_1.hds",
        original_dir / "GWF_1/disv.disv.grb",
    )

    # TODO: Fix issue 669
    #    original_cbc = imod.mf6.open_cbc(
    #        original_dir / "GWF_1/GWF_1.cbc",
    #        original_dir / "GWF_1/disv.disv.grb",
    #    )

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    # _ = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the partitioned simulation
    np.testing.assert_allclose(
        head["head"].values, original_head.values, rtol=1e-5, atol=1e-3
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

    original_head = imod.mf6.open_hds(
        original_dir / "GWF_1/GWF_1.hds",
        original_dir / "GWF_1/disv.disv.grb",
    )

    original_cbc = imod.mf6.open_cbc(
        original_dir / "GWF_1/GWF_1.cbc",
        original_dir / "GWF_1/disv.disv.grb",
    )

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()

    cbc = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the
    # partitioned simulation. Criteria are a bit looser than in other tests
    # because we are dealing with a problem with heads ranging roughly from 2000
    # m to 0 m, and the HFB adds extra complexity to this.
    np.testing.assert_allclose(head["head"].values, original_head.values, rtol=1e-4)
    np.testing.assert_allclose(cbc["chd"].values, original_cbc["chd"].values, rtol=1e-4)


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

    original_head = imod.mf6.open_hds(
        original_dir / "GWF_1/GWF_1.hds",
        original_dir / "GWF_1/disv.disv.grb",
    )

    # TODO:
    # Uncomment when fixed: https://gitlab.com/deltares/imod/imod-python/-/issues/683
    # original_cbc = imod.mf6.open_cbc(
    #     original_dir / "GWF_1/GWF_1.cbc",
    #     original_dir / "GWF_1/disv.disv.grb",
    # )

    # Partition the simulation, run it, and save the (merged) results
    split_simulation = simulation.split(partition_array)

    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    head = split_simulation.open_head()
    # TODO:
    # Uncomment when fixed: https://gitlab.com/deltares/imod/imod-python/-/issues/683
    # cbc = split_simulation.open_flow_budget()

    # Compare the head result of the original simulation with the result of the
    # partitioned simulation.
    np.testing.assert_allclose(
        head["head"].values, original_head.values, rtol=1e-5, atol=1e-3
    )
    # TODO:
    # Uncomment when fixed: https://gitlab.com/deltares/imod/imod-python/-/issues/683
    # np.testing.assert_allclose(
    #     cbc["chd"].values, original_cbc["chd"].values, rtol=1e-5, atol=1e-3
    # )

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

    split_head.ugrid.grid.node_x = split_head.ugrid.grid.node_x.round(5)
    split_head.ugrid.grid.node_y = split_head.ugrid.grid.node_y.round(5)    
    original_heads.ugrid.grid.node_x = original_heads.ugrid.grid.node_x.round(5)
    original_heads.ugrid.grid.node_y = original_heads.ugrid.grid.node_y.round(5)
    split_head = split_head.ugrid.reindex_like(original_heads)


    split_balances["npf-qx"] .ugrid.grid.node_x = split_balances["npf-qx"] .ugrid.grid.node_x.round(5)
    split_balances["npf-qx"] .ugrid.grid.node_y = split_balances["npf-qx"] .ugrid.grid.node_y.round(5)
    original_balances["npf-qx"].ugrid.grid.node_x = original_balances["npf-qx"].ugrid.grid.node_x.round(5)        
    original_balances["npf-qx"].ugrid.grid.node_y = original_balances["npf-qx"].ugrid.grid.node_y.round(5)

    split_balances["npf-qy"] .ugrid.grid.node_x = split_balances["npf-qy"] .ugrid.grid.node_x.round(5)
    split_balances["npf-qy"] .ugrid.grid.node_y = split_balances["npf-qy"] .ugrid.grid.node_y.round(5)
    original_balances["npf-qy"].ugrid.grid.node_x = original_balances["npf-qy"].ugrid.grid.node_x.round(5)        
    original_balances["npf-qy"].ugrid.grid.node_y = original_balances["npf-qy"].ugrid.grid.node_y.round(5)

    split_balances["npf-qx"].ugrid.to_netcdf(tmp_path / "split_balances_x.nc")
    split_balances["npf-qy"].ugrid.to_netcdf(tmp_path / "split_balances_y.nc")    
    original_balances["npf-qx"].ugrid.to_netcdf(tmp_path / "original_balances_x.nc")
    original_balances["npf-qy"].ugrid.to_netcdf(tmp_path / "original_balances_y.nc")
 
    split_balances_x_v2 = xu.open_dataset(tmp_path /  "split_balances_x.nc")
    split_balances_y_v2 = xu.open_dataset(tmp_path /  "split_balances_y.nc")    
    original_balances_x_v2 =  xu.open_dataset(tmp_path /  "original_balances_x.nc")
    original_balances_y_v2 =  xu.open_dataset(tmp_path /  "original_balances_y.nc")


    split_balances_x_v2["npf-qx"] = split_balances_x_v2["npf-qx"].ugrid.reindex_like( original_balances_x_v2)
    split_balances_y_v2["npf-qy"] = split_balances_y_v2["npf-qy"].ugrid.reindex_like( original_balances_y_v2 )
    head_diff = original_heads.isel(layer=0, time=-1) - split_head.isel(layer = 0, time =-1)

    veldif_x = original_balances_x_v2["data-spdis"] - split_balances_x_v2["npf-qx"]
    veldif_y = original_balances_y_v2 ["data-spdis"]- split_balances_y_v2["npf-qy"]

    assert veldif_x.values.max() < 1e-6
    assert veldif_y.values.max() < 1e-6

    reldif_x = abs(veldif_x / original_balances_x_v2)
    reldif_y = abs(veldif_y / original_balances_y_v2)

    mask_x = abs(original_balances_x_v2["data-spdis"])  < 1e-15
    mask_y = abs(original_balances_y_v2["data-spdis"])  < 1e-15    
    reldif_x = xr.where(mask_x, 0, reldif_x["data-spdis"])
    reldif_y =  xr.where(mask_y, 0, reldif_y["data-spdis"])
    print(f"reldif x: {reldif_x.max() }")
    print(f"reldif y: {reldif_y.max() }")
    assert reldif_x.max() < 1e-5
    assert reldif_y.max() < 0.12


    pass
