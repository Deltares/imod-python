import numpy as np
import pytest
import xarray as xr
import xugrid as xu
from pytest_cases import parametrize_with_cases

from imod.prepare.topsystem import (
    ALLOCATION_OPTION,
    DISTRIBUTING_OPTION,
    allocate_drn_cells,
    allocate_ghb_cells,
    allocate_rch_cells,
    allocate_riv_cells,
    distribute_drn_conductance,
    distribute_ghb_conductance,
    distribute_riv_conductance,
)
from imod.typing import GridDataArray
from imod.typing.grid import is_unstructured, zeros_like
from imod.util.dims import enforce_dim_order


# TODO: Move to flow_basic_fixture.py
def make_basic_dis(dz, nrow, ncol):
    """Basic model discretization"""

    dx = 10.0
    dy = -10.0

    nlay = len(dz)

    shape = nlay, nrow, ncol

    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    ibound = xr.DataArray(np.ones(shape, dtype=np.int32), coords=coords, dims=dims)

    surface = 0.0
    interfaces = np.insert((surface - np.cumsum(dz)), 0, surface)

    bottom = xr.DataArray(interfaces[1:], coords={"layer": layer}, dims="layer")
    top = xr.DataArray(interfaces[:-1], coords={"layer": layer}, dims="layer")

    return ibound, top, bottom


@pytest.fixture(scope="function")
def basic_dis_riv():
    return make_basic_dis(dz=[1., 2., 4., 10.], nrow=9, ncol=9)

@pytest.fixture(scope="function")
def basic_disv_riv():
    ibound, top, bottom = make_basic_dis(dz=[1., 2., 4., 10.], nrow=9, ncol=9)
    idomain_ugrid = xu.UgridDataArray.from_structured(ibound)

    return idomain_ugrid, top, bottom


def take_first_planar_cell(grid: GridDataArray):
    if is_unstructured(grid):
        return grid.values[:, 0]
    else:
        return grid.values[:, 0, 0]

#########
# CASES #
#########

class RiverCases:
    def case_structured(self, basic_dis_riv):
        ibound, top, bottom = basic_dis_riv
        top = top.sel(layer=1)
        elevation = zeros_like(ibound.sel(layer=1))
        stage = elevation - 2.0
        bottom_elevation = elevation - 4.0
        active = ibound == 1
        return active, top, bottom, stage, bottom_elevation

    def case_unstructured(self, basic_disv_riv):
        ibound, top, bottom = basic_disv_riv
        elevation = zeros_like(ibound.sel(layer=1))
        stage = elevation - 2.0
        bottom_elevation = elevation - 4.0
        active = ibound == 1
        return active, top, bottom, stage, bottom_elevation


class DrainCases:
    def case_structured(self, basic_dis_riv):
        ibound, top, bottom = basic_dis_riv
        top = top.sel(layer=1)
        elevation = zeros_like(ibound.sel(layer=1))
        drain_elevation = elevation - 2.0
        active = ibound == 1
        return active, top, bottom, drain_elevation

    def case_unstructured(self, basic_disv_riv):
        ibound, top, bottom = basic_disv_riv
        elevation = zeros_like(ibound.sel(layer=1))
        drain_elevation = elevation - 2.0
        active = ibound == 1
        return active, top, bottom, drain_elevation


class GeneralHeadBoundaryCases:
    def case_structured(self, basic_dis_riv):
        ibound, top, bottom = basic_dis_riv
        top = top.sel(layer=1)
        elevation = zeros_like(ibound.sel(layer=1))
        head = elevation - 2.0
        active = ibound == 1
        return active, top, bottom, head

    def case_unstructured(self, basic_disv_riv):
        ibound, top, bottom = basic_disv_riv
        elevation = zeros_like(ibound.sel(layer=1))
        head = elevation - 2.0
        active = ibound == 1
        return active, top, bottom, head


class RechargeCases:
    def case_structured(self, basic_dis_riv):
        ibound, _, _ = basic_dis_riv
        active = ibound == 1
        return active

    def case_unstructured(self, basic_disv_riv):
        ibound, _, _ = basic_disv_riv
        active = ibound == 1
        return active


class AllocationOptionRiverCases:
    def case_stage_to_riv_bot(self):
        option = ALLOCATION_OPTION.stage_to_riv_bot
        expected = [False, True, True, False]

        return option, expected, None

    def case_first_active_to_riv_bot(self):
        option = ALLOCATION_OPTION.first_active_to_riv_bot
        expected = [True, True, True, False]

        return option, expected, None

    def case_first_active_to_riv_bot__drn(self):
        option = ALLOCATION_OPTION.first_active_to_riv_bot__drn
        expected = [False, True, True, False]
        expected__drn = [True, False, False, False]

        return option, expected, expected__drn

    def case_at_elevation(self):
        option = ALLOCATION_OPTION.at_elevation
        expected = [False, True, False, False]

        return option, expected, None

    def case_at_first_active(self):
        option = ALLOCATION_OPTION.at_first_active
        expected = [True, False, False, False]

        return option, expected, None


class AllocationOptionDrainCases:
    def case_at_elevation(self):
        option = ALLOCATION_OPTION.at_elevation
        expected = [False, True, False, False]

        return option, expected

    def case_at_first_active(self):
        option = ALLOCATION_OPTION.at_first_active
        expected = [True, False, False, False]

        return option, expected


class AllocationOptionGeneralHeadCases:
    def case_at_elevation(self):
        option = ALLOCATION_OPTION.at_elevation
        expected = [False, True, False, False]

        return option, expected

    def case_at_first_active(self):
        option = ALLOCATION_OPTION.at_first_active
        expected = [True, False, False, False]

        return option, expected


class AllocationOptionRechargeCases:
    def case_at_first_active(self):
        option = ALLOCATION_OPTION.at_first_active
        expected = [True, False, False, False]

        return option, expected


class DistributionOptionRiverCases:
    def case_by_corrected_transmissivity(self):
        option = DISTRIBUTING_OPTION.by_corrected_transmissivity
        allocated_layer = xr.DataArray([False, True, True, False], coords={"layer": [1,2,3,4]}, dims=("layer",))
        expected = [np.nan, (4/5), (1/5), np.nan]
        return option, allocated_layer, expected

    def case_by_crosscut_transmissivity(self):
        option = DISTRIBUTING_OPTION.by_crosscut_transmissivity
        allocated_layer = xr.DataArray([False, True, True, False], coords={"layer": [1,2,3,4]}, dims=("layer",))
        expected = [np.nan, (2/3), (1/3), np.nan]
        return option, allocated_layer, expected

    def case_by_crosscut_thickness(self):
        option = DISTRIBUTING_OPTION.by_crosscut_thickness
        allocated_layer = xr.DataArray([False, True, True, False], coords={"layer": [1,2,3,4]}, dims=("layer",))
        expected = [np.nan, 0.5, 0.5, np.nan]
        return option, allocated_layer, expected

    def case_by_conductivity(self):
        option = DISTRIBUTING_OPTION.by_conductivity
        allocated_layer = xr.DataArray([False, True, True, False], coords={"layer": [1,2,3,4]}, dims=("layer",))
        expected = [np.nan, (2/3), (1/3), np.nan]
        return option, allocated_layer, expected

    def case_by_layer_thickness(self):
        option = DISTRIBUTING_OPTION.by_layer_thickness
        allocated_layer = xr.DataArray([False, True, True, False], coords={"layer": [1,2,3,4]}, dims=("layer",))
        expected = [np.nan, (1/3), (2/3), np.nan]
        return option, allocated_layer, expected

    def case_by_layer_transmissivity(self):
        option = DISTRIBUTING_OPTION.by_layer_transmissivity
        allocated_layer = xr.DataArray([False, True, True, False], coords={"layer": [1,2,3,4]}, dims=("layer",))
        expected = [np.nan, 0.5, 0.5, np.nan]
        return option, allocated_layer, expected

    def case_equally(self):
        option = DISTRIBUTING_OPTION.equally
        allocated_layer = xr.DataArray([False, True, True, False], coords={"layer": [1,2,3,4]}, dims=("layer",))
        expected = [np.nan, 0.5, 0.5, np.nan]
        return option, allocated_layer, expected


#########
# TESTS #
#########

@parametrize_with_cases(
    argnames="active,top,bottom,stage,bottom_elevation",
    cases=RiverCases,
)
@parametrize_with_cases(
    argnames="option,expected_riv,expected_drn", cases=AllocationOptionRiverCases
)
def test_riv_allocation(
    active, top, bottom, stage, bottom_elevation, option, expected_riv, expected_drn
):
    actual_riv_da, actual_drn_da = allocate_riv_cells(
        option, active, top, bottom, stage, bottom_elevation
    )

    actual_riv = take_first_planar_cell(actual_riv_da)

    if actual_drn_da is None:
        actual_drn = actual_drn_da
    else:
        actual_drn = take_first_planar_cell(actual_drn_da)

    np.testing.assert_equal(actual_riv, expected_riv)
    np.testing.assert_equal(actual_drn, expected_drn)


@parametrize_with_cases(
    argnames="active,top,bottom,drn_elevation",
    cases=DrainCases,
)
@parametrize_with_cases(argnames="option,expected", cases=AllocationOptionDrainCases)
def test_drn_allocation(active, top, bottom, drn_elevation, option, expected):
    actual_da = allocate_drn_cells(option, active, top, bottom, drn_elevation)

    actual = take_first_planar_cell(actual_da)

    np.testing.assert_equal(actual, expected)


@parametrize_with_cases(
    argnames="active,top,bottom,head",
    cases=GeneralHeadBoundaryCases,
)
@parametrize_with_cases(
    argnames="option,expected", cases=AllocationOptionGeneralHeadCases
)
def test_ghb_allocation(active, top, bottom, head, option, expected):
    actual_da = allocate_ghb_cells(option, active, top, bottom, head)

    actual = take_first_planar_cell(actual_da)

    np.testing.assert_equal(actual, expected)


@parametrize_with_cases(
    argnames="active",
    cases=RechargeCases,
)
@parametrize_with_cases(argnames="option,expected", cases=AllocationOptionRechargeCases)
def test_rch_allocation(active, option, expected):
    actual_da = allocate_rch_cells(option, active)

    actual = take_first_planar_cell(actual_da)

    np.testing.assert_equal(actual, expected)


@parametrize_with_cases(
    argnames="active,top,bottom,stage,bottom_elevation",
    cases=RiverCases,
)
@parametrize_with_cases(
    argnames="option,allocated_layer,expected",
    cases=DistributionOptionRiverCases,
)
def test_distribute_riv_conductance(active, top, bottom, stage, bottom_elevation, option, allocated_layer, expected):
    allocated = enforce_dim_order(active & allocated_layer)
    k = xr.DataArray([2.0, 2.0, 1.0, 1.0], coords={"layer": [1,2,3,4]}, dims=("layer",))

    conductance = zeros_like(bottom_elevation) + 1.0

    actual_da = distribute_riv_conductance(option, allocated, conductance, top, bottom, stage, bottom_elevation, k)
    actual = take_first_planar_cell(actual_da)

    np.testing.assert_equal(actual, expected)