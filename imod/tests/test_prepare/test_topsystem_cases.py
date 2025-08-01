import numpy as np
import xarray as xr
from pytest_cases import case

from imod.prepare.topsystem import (
    ALLOCATION_OPTION,
    DISTRIBUTING_OPTION,
)
from imod.typing.grid import zeros_like


def time_da():
    time_ls = [
        np.datetime64(t)
        for t in [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ]
    ]
    return xr.DataArray([1.0, 1.0, 1.0], coords={"time": time_ls}, dims=("time",))


def riv_structured_transient(basic_dis__topsystem):
    ibound, top, bottom = basic_dis__topsystem
    top = top.sel(layer=1, drop=True)
    elevation = zeros_like(ibound.sel(layer=1), dtype=np.float64)
    # Deactivate second cell
    elevation[1, 1] = np.nan
    elevation = (elevation * time_da()).transpose("time", "y", "x")
    stage = elevation - 2.0
    bottom_elevation = elevation - 4.0
    active = ibound == 1
    return active, top, bottom, stage, bottom_elevation


def riv_unstructured_transient(basic_disv__topsystem):
    ibound, top, bottom = basic_disv__topsystem
    top = top.sel(layer=1, drop=True)
    elevation = zeros_like(ibound.sel(layer=1), dtype=np.float64)
    # Deactivate second cell
    elevation[1] = np.nan
    face_dim = elevation.ugrid.grid.face_dimension
    elevation = (elevation * time_da()).transpose("time", face_dim)
    stage = elevation - 2.0
    bottom_elevation = elevation - 4.0
    active = ibound == 1
    return active, top, bottom, stage, bottom_elevation


def riv_structured(basic_dis__topsystem):
    ibound, top, bottom = basic_dis__topsystem
    top = top.sel(layer=1, drop=True)
    elevation = zeros_like(ibound.sel(layer=1), dtype=np.float64)
    # Deactivate second cell
    elevation[1, 1] = np.nan
    stage = elevation - 2.0
    bottom_elevation = elevation - 4.0
    active = ibound == 1
    return active, top, bottom, stage, bottom_elevation


def riv_unstructured(basic_disv__topsystem):
    ibound, top, bottom = basic_disv__topsystem
    top = top.sel(layer=1, drop=True)
    elevation = zeros_like(ibound.sel(layer=1), dtype=np.float64)
    # Deactivate second cell
    elevation[1] = np.nan
    stage = elevation - 2.0
    bottom_elevation = elevation - 4.0
    active = ibound == 1
    return active, top, bottom, stage, bottom_elevation


def drn_structured(basic_dis__topsystem):
    ibound, top, bottom = basic_dis__topsystem
    top = top.sel(layer=1, drop=True)
    elevation = zeros_like(ibound.sel(layer=1), dtype=np.float64)
    # Deactivate second cell
    elevation[1, 1] = np.nan
    drain_elevation = elevation - 4.0
    active = ibound == 1
    return active, top, bottom, drain_elevation


def drn_unstructured(basic_disv__topsystem):
    ibound, top, bottom = basic_disv__topsystem
    top = top.sel(layer=1, drop=True)
    elevation = zeros_like(ibound.sel(layer=1), dtype=np.float64)
    # Deactivate second cell
    elevation[1] = np.nan
    drain_elevation = elevation - 4.0
    active = ibound == 1
    return active, top, bottom, drain_elevation


def ghb_structured(basic_dis__topsystem):
    ibound, top, bottom = basic_dis__topsystem
    top = top.sel(layer=1, drop=True)
    elevation = zeros_like(ibound.sel(layer=1), dtype=np.float64)
    # Deactivate second cell
    elevation[1, 1] = np.nan
    head = elevation - 4.0
    active = ibound == 1
    return active, top, bottom, head


def ghb_unstructured(basic_disv__topsystem):
    ibound, top, bottom = basic_disv__topsystem
    top = top.sel(layer=1, drop=True)
    elevation = zeros_like(ibound.sel(layer=1), dtype=np.float64)
    # Deactivate second cell
    elevation[1] = np.nan
    head = elevation - 4.0
    active = ibound == 1
    return active, top, bottom, head


def rch_structured(basic_dis__topsystem):
    ibound, _, _ = basic_dis__topsystem
    elevation = zeros_like(ibound.sel(layer=1), dtype=np.float64)
    rate = elevation + 0.001
    # Deactivate second cell
    rate[1, 1] = np.nan
    active = ibound == 1
    return active, rate


def rch_unstructured(basic_disv__topsystem):
    ibound, _, _ = basic_disv__topsystem
    elevation = zeros_like(ibound.sel(layer=1), dtype=np.float64)
    rate = elevation + 0.001
    # Deactivate second cell
    rate[1] = np.nan
    active = ibound == 1
    return active, rate


@case(tags=["riv"])
def allocation_stage_to_riv_bot():
    option = ALLOCATION_OPTION.stage_to_riv_bot
    expected = [False, True, True, False]

    return option, expected, None


@case(tags=["riv", "drn", "ghb"])
def allocation_first_active_to_elevation():
    option = ALLOCATION_OPTION.first_active_to_elevation
    expected = [True, True, True, False]

    return option, expected, None


@case(tags=["riv"])
def allocation_stage_to_riv_bot_drn_above():
    option = ALLOCATION_OPTION.stage_to_riv_bot_drn_above
    expected = [False, True, True, False]
    expected__drn = [True, False, False, False]

    return option, expected, expected__drn


@case(tags=["drn", "ghb", "riv"])
def allocation_at_elevation():
    option = ALLOCATION_OPTION.at_elevation
    expected = [False, False, True, False]

    return option, expected, None


@case(tags=["riv", "drn", "ghb", "rch"])
def allocation_at_first_active():
    option = ALLOCATION_OPTION.at_first_active
    expected = [True, False, False, False]

    return option, expected, None


@case(tags=["riv"])
def distribution_by_corrected_transmissivity():
    option = DISTRIBUTING_OPTION.by_corrected_transmissivity
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (4 / 5), (1 / 5), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_corrected_transmissivity__first_active():
    """First active cell allocated, while stage in second layer"""
    option = DISTRIBUTING_OPTION.by_corrected_transmissivity
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (4 / 5), (1 / 5), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn"])
def distribution_by_corrected_transmissivity__third_only():
    """Third layer active only, while stage in second layer"""
    option = DISTRIBUTING_OPTION.by_corrected_transmissivity
    allocated_layer = xr.DataArray(
        [False, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_corrected_transmissivity__TFTF():
    option = DISTRIBUTING_OPTION.by_corrected_transmissivity
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_corrected_transmissivity__drn():
    option = DISTRIBUTING_OPTION.by_corrected_transmissivity
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (16 / 17), (1 / 17), np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_corrected_transmissivity__first_active__drn():
    """First active cell allocated, while drain in third layer"""
    option = DISTRIBUTING_OPTION.by_corrected_transmissivity
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(8 / 25), (16 / 25), (1 / 25), np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_corrected_transmissivity__TFTF__drn():
    option = DISTRIBUTING_OPTION.by_corrected_transmissivity
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(8 / 9), np.nan, (1 / 9), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_corrected_thickness():
    option = DISTRIBUTING_OPTION.by_corrected_thickness
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (2 / 3), (1 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_corrected_thickness__first_active():
    """First active cell allocated, while stage in second layer"""
    option = DISTRIBUTING_OPTION.by_corrected_thickness
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (2 / 3), (1 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn"])
def distribution_by_corrected_thickness__third_only():
    """Third layer active only, while stage in second layer"""
    option = DISTRIBUTING_OPTION.by_corrected_thickness
    allocated_layer = xr.DataArray(
        [False, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_corrected_thickness__TFTF():
    option = DISTRIBUTING_OPTION.by_corrected_thickness
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_corrected_thickness__drn():
    option = DISTRIBUTING_OPTION.by_corrected_thickness
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (8 / 9), (1 / 9), np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_corrected_thickness__first_active__drn():
    """First active cell allocated, while drain in third layer"""
    option = DISTRIBUTING_OPTION.by_corrected_thickness
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(4 / 13), (8 / 13), (1 / 13), np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_corrected_thickness__TFTF__drn():
    option = DISTRIBUTING_OPTION.by_corrected_thickness
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(4 / 5), np.nan, (1 / 5), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_crosscut_transmissivity():
    option = DISTRIBUTING_OPTION.by_crosscut_transmissivity
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (2 / 3), (1 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_crosscut_transmissivity__first_active():
    """First active cell allocated, while stage in second layer"""
    option = DISTRIBUTING_OPTION.by_crosscut_transmissivity
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (2 / 3), (1 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn"])
def distribution_by_crosscut_transmissivity__third_only():
    """Third layer active only, while stage in second layer"""
    option = DISTRIBUTING_OPTION.by_crosscut_transmissivity
    allocated_layer = xr.DataArray(
        [False, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_crosscut_transmissivity__TFTF():
    """Third layer active only, while stage in second layer"""
    option = DISTRIBUTING_OPTION.by_crosscut_transmissivity
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_crosscut_transmissivity__drn():
    option = DISTRIBUTING_OPTION.by_crosscut_transmissivity
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (4 / 5), (1 / 5), np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_crosscut_transmissivity__first_active__drn():
    """First active cell allocated, while drain elevation in third layer"""
    option = DISTRIBUTING_OPTION.by_crosscut_transmissivity
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(2 / 7), (4 / 7), (1 / 7), np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_crosscut_transmissivity__TFTF__drn():
    """First and third layer active, while drain elevation in third layer"""
    option = DISTRIBUTING_OPTION.by_crosscut_transmissivity
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(2 / 3), np.nan, (1 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_crosscut_thickness():
    option = DISTRIBUTING_OPTION.by_crosscut_thickness
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, 0.5, 0.5, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_crosscut_thickness__first_active():
    """First active cell allocated, while stage in second layer"""
    option = DISTRIBUTING_OPTION.by_crosscut_thickness
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, 0.5, 0.5, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn"])
def distribution_by_crosscut_thickness__third_only():
    """Third layer active only, while stage in second layer"""
    option = DISTRIBUTING_OPTION.by_crosscut_thickness
    allocated_layer = xr.DataArray(
        [False, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_crosscut_thickness__TFTF():
    """Third layer active only, while stage in second layer"""
    option = DISTRIBUTING_OPTION.by_crosscut_thickness
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_crosscut_thickness__drn():
    option = DISTRIBUTING_OPTION.by_crosscut_thickness
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (2 / 3), (1 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_crosscut_thickness__first_active__drn():
    """First active cell allocated, while drain elevation in second layer"""
    option = DISTRIBUTING_OPTION.by_crosscut_thickness
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [0.25, 0.5, 0.25, np.nan]
    return option, allocated_layer, expected


@case(tags=["drn"])
def distribution_by_crosscut_thickness__TFTF__drn():
    """First and third layer active, while drain elevation in second layer"""
    option = DISTRIBUTING_OPTION.by_crosscut_thickness
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [0.5, np.nan, 0.5, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_conductivity():
    option = DISTRIBUTING_OPTION.by_conductivity
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (2 / 3), (1 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_conductivity__first_active():
    option = DISTRIBUTING_OPTION.by_conductivity
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(2 / 5), (2 / 5), (1 / 5), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_conductivity__third_only():
    option = DISTRIBUTING_OPTION.by_conductivity
    allocated_layer = xr.DataArray(
        [False, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_conductivity__TFTF():
    option = DISTRIBUTING_OPTION.by_conductivity
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(2 / 3), np.nan, (1 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_layer_thickness():
    option = DISTRIBUTING_OPTION.by_layer_thickness
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (1 / 3), (2 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_layer_thickness__first_active():
    option = DISTRIBUTING_OPTION.by_layer_thickness
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(1 / 7), (2 / 7), (4 / 7), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_layer_thickness__third_only():
    option = DISTRIBUTING_OPTION.by_layer_thickness
    allocated_layer = xr.DataArray(
        [False, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_layer_thickness__TFTF():
    option = DISTRIBUTING_OPTION.by_layer_thickness
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(1 / 5), np.nan, (4 / 5), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_layer_transmissivity():
    option = DISTRIBUTING_OPTION.by_layer_transmissivity
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, 0.5, 0.5, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_layer_transmissivity__first_active():
    option = DISTRIBUTING_OPTION.by_layer_transmissivity
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [0.2, 0.4, 0.4, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_layer_transmissivity__third_only():
    option = DISTRIBUTING_OPTION.by_layer_transmissivity
    allocated_layer = xr.DataArray(
        [False, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_by_layer_transmissivity__TFTF():
    option = DISTRIBUTING_OPTION.by_layer_transmissivity
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(1 / 3), np.nan, (2 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_equally():
    option = DISTRIBUTING_OPTION.equally
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, 0.5, 0.5, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_equally__first_active():
    option = DISTRIBUTING_OPTION.equally
    allocated_layer = xr.DataArray(
        [True, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [(1 / 3), (1 / 3), (1 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_equally__third_only():
    option = DISTRIBUTING_OPTION.equally
    allocated_layer = xr.DataArray(
        [False, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, np.nan, 1.0, np.nan]
    return option, allocated_layer, expected


@case(tags=["riv", "drn", "ghb"])
def distribution_equally__TFTF():
    option = DISTRIBUTING_OPTION.equally
    allocated_layer = xr.DataArray(
        [True, False, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [0.5, np.nan, 0.5, np.nan]
    return option, allocated_layer, expected
