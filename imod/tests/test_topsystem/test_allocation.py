import numpy as np
from pytest_cases import parametrize_with_cases

from imod.prepare.topsystem import (
    ALLOCATION_OPTION,
    allocate_drain_cells,
    allocate_ghb_cells,
    allocate_rch_cells,
    allocate_river_cells,
)
from imod.typing import GridDataArray
from imod.typing.grid import is_unstructured, zeros_like


def take_first_planar_cell(grid: GridDataArray):
    if is_unstructured(grid):
        return grid.values[:, 0]
    else:
        return grid.values[:, 0, 0]


class RiverCases:
    def case_structured(self, basic_dis):
        ibound, top, bottom = basic_dis
        top = top.sel(layer=1)
        elevation = zeros_like(ibound.sel(layer=1))
        stage = elevation - 7.5
        bottom_elevation = elevation - 10.0
        active = ibound == 1
        return active, top, bottom, stage, bottom_elevation

    def case_unstructured(self, basic_unstructured_dis):
        ibound, top, bottom = basic_unstructured_dis
        elevation = zeros_like(ibound.sel(layer=1))
        stage = elevation - 7.5
        bottom_elevation = elevation - 10.0
        active = ibound == 1
        return active, top, bottom, stage, bottom_elevation


class DrainCases:
    def case_structured(self, basic_dis):
        ibound, top, bottom = basic_dis
        top = top.sel(layer=1)
        elevation = zeros_like(ibound.sel(layer=1))
        drain_elevation = elevation - 7.5
        active = ibound == 1
        return active, top, bottom, drain_elevation

    def case_unstructured(self, basic_unstructured_dis):
        ibound, top, bottom = basic_unstructured_dis
        elevation = zeros_like(ibound.sel(layer=1))
        drain_elevation = elevation - 7.5
        active = ibound == 1
        return active, top, bottom, drain_elevation


class GeneralHeadBoundaryCases:
    def case_structured(self, basic_dis):
        ibound, top, bottom = basic_dis
        top = top.sel(layer=1)
        elevation = zeros_like(ibound.sel(layer=1))
        head = elevation - 7.5
        active = ibound == 1
        return active, top, bottom, head

    def case_unstructured(self, basic_unstructured_dis):
        ibound, top, bottom = basic_unstructured_dis
        elevation = zeros_like(ibound.sel(layer=1))
        head = elevation - 7.5
        active = ibound == 1
        return active, top, bottom, head


class RechargeCases:
    def case_structured(self, basic_dis):
        ibound, _, _ = basic_dis
        active = ibound == 1
        return active

    def case_unstructured(self, basic_unstructured_dis):
        ibound, _, _ = basic_unstructured_dis
        active = ibound == 1
        return active


class AllocationOptionRiverCases:
    def case_stage_to_riv_bot(self):
        option = ALLOCATION_OPTION.stage_to_riv_bot
        expected = [False, True, False]

        return option, expected, None

    def case_first_active_to_riv_bot(self):
        option = ALLOCATION_OPTION.first_active_to_riv_bot
        expected = [True, True, False]

        return option, expected, None

    def case_first_active_to_riv_bot__drn(self):
        option = ALLOCATION_OPTION.first_active_to_riv_bot__drn
        expected = [False, True, False]
        expected__drn = [True, False, False]

        return option, expected, expected__drn

    def case_at_elevation(self):
        option = ALLOCATION_OPTION.at_elevation
        expected = [False, True, False]

        return option, expected, None

    def case_at_first_active(self):
        option = ALLOCATION_OPTION.at_first_active
        expected = [True, False, False]

        return option, expected, None


class AllocationOptionDrainCases:
    def case_at_elevation(self):
        option = ALLOCATION_OPTION.at_elevation
        expected = [False, True, False]

        return option, expected

    def case_at_first_active(self):
        option = ALLOCATION_OPTION.at_first_active
        expected = [True, False, False]

        return option, expected


class AllocationOptionGeneralHeadCases:
    def case_at_elevation(self):
        option = ALLOCATION_OPTION.at_elevation
        expected = [False, True, False]

        return option, expected

    def case_at_first_active(self):
        option = ALLOCATION_OPTION.at_first_active
        expected = [True, False, False]

        return option, expected


class AllocationOptionRechargeCases:
    def case_at_first_active(self):
        option = ALLOCATION_OPTION.at_first_active
        expected = [True, False, False]

        return option, expected


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
    actual_riv_da, actual_drn_da = allocate_river_cells(
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
    actual_da = allocate_drain_cells(option, active, top, bottom, drn_elevation)

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
