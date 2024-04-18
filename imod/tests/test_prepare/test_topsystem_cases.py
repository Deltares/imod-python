import numpy as np
import xarray as xr
from pytest_cases import case

from imod.prepare.topsystem import (
    ALLOCATION_OPTION,
    DISTRIBUTING_OPTION,
)
from imod.typing.grid import zeros_like


def riv_structured(basic_dis__topsystem):
    ibound, top, bottom = basic_dis__topsystem
    top = top.sel(layer=1)
    elevation = zeros_like(ibound.sel(layer=1))
    stage = elevation - 2.0
    bottom_elevation = elevation - 4.0
    active = ibound == 1
    return active, top, bottom, stage, bottom_elevation


def riv_unstructured(basic_disv__topsystem):
    ibound, top, bottom = basic_disv__topsystem
    elevation = zeros_like(ibound.sel(layer=1))
    stage = elevation - 2.0
    bottom_elevation = elevation - 4.0
    active = ibound == 1
    return active, top, bottom, stage, bottom_elevation


def drn_structured(basic_dis__topsystem):
    ibound, top, bottom = basic_dis__topsystem
    top = top.sel(layer=1)
    elevation = zeros_like(ibound.sel(layer=1))
    drain_elevation = elevation - 2.0
    active = ibound == 1
    return active, top, bottom, drain_elevation


def drn_unstructured(basic_disv__topsystem):
    ibound, top, bottom = basic_disv__topsystem
    elevation = zeros_like(ibound.sel(layer=1))
    drain_elevation = elevation - 2.0
    active = ibound == 1
    return active, top, bottom, drain_elevation


def ghb_structured(basic_dis__topsystem):
    ibound, top, bottom = basic_dis__topsystem
    top = top.sel(layer=1)
    elevation = zeros_like(ibound.sel(layer=1))
    head = elevation - 2.0
    active = ibound == 1
    return active, top, bottom, head


def ghb_unstructured(basic_disv__topsystem):
    ibound, top, bottom = basic_disv__topsystem
    elevation = zeros_like(ibound.sel(layer=1))
    head = elevation - 2.0
    active = ibound == 1
    return active, top, bottom, head


def rch_structured(basic_dis__topsystem):
    ibound, _, _ = basic_dis__topsystem
    active = ibound == 1
    return active


def rch_unstructured(basic_disv__topsystem):
    ibound, _, _ = basic_disv__topsystem
    active = ibound == 1
    return active


@case(tags=["riv"])
def allocation_stage_to_riv_bot():
    option = ALLOCATION_OPTION.stage_to_riv_bot
    expected = [False, True, True, False]

    return option, expected, None


@case(tags=["riv"])
def allocation_first_active_to_riv_bot():
    option = ALLOCATION_OPTION.first_active_to_riv_bot
    expected = [True, True, True, False]

    return option, expected, None


@case(tags=["riv"])
def allocation_first_active_to_riv_bot__drn():
    option = ALLOCATION_OPTION.first_active_to_riv_bot__drn
    expected = [False, True, True, False]
    expected__drn = [True, False, False, False]

    return option, expected, expected__drn


@case(tags=["drn", "ghb"])
def allocation_at_elevation():
    option = ALLOCATION_OPTION.at_elevation
    expected = [False, True, False, False]

    return option, expected, None


@case(tags=["riv"])
def allocation_at_riv_bottom_elevation():
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
def distribution_by_crosscut_transmissivity():
    option = DISTRIBUTING_OPTION.by_crosscut_transmissivity
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (2 / 3), (1 / 3), np.nan]
    return option, allocated_layer, expected


@case(tags=["riv"])
def distribution_by_crosscut_thickness():
    option = DISTRIBUTING_OPTION.by_crosscut_thickness
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, 0.5, 0.5, np.nan]
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
def distribution_by_layer_thickness():
    option = DISTRIBUTING_OPTION.by_layer_thickness
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, (1 / 3), (2 / 3), np.nan]
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
def distribution_equally():
    option = DISTRIBUTING_OPTION.equally
    allocated_layer = xr.DataArray(
        [False, True, True, False], coords={"layer": [1, 2, 3, 4]}, dims=("layer",)
    )
    expected = [np.nan, 0.5, 0.5, np.nan]
    return option, allocated_layer, expected