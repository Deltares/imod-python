import numpy as np
from pytest_cases import parametrize_with_cases

from imod.prepare.topsystem import ALLOCATION_OPTION, allocate_river_cells
from imod.typing.grid import zeros_like


class RiverCases:
    def case_structured(basic_dis):
        ibound, top, bottom = basic_dis
        top = top.sel(layer=1)
        elevation = zeros_like(ibound.sel(layer=1))
        stage = elevation - 2.5
        bottom_elevation = elevation - 10.0
        active = ibound == 1
        return active, top, bottom, stage, bottom_elevation

    def case_unstructured(basic_unstructured_dis):
        ibound, top, bottom = basic_unstructured_dis
        elevation = zeros_like(ibound.sel(layer=1))
        stage = elevation - 2.5
        bottom_elevation = elevation - 10.0
        active = ibound == 1
        return active, top, bottom, stage, bottom_elevation


class AllocationOptionCases:
    def case_stage_to_riv_bot():
        option = ALLOCATION_OPTION.stage_to_riv_bot
        expected = [True, True, False]

        return option, expected


@parametrize_with_cases(argnames=["active,top,bottom,stage,bottom_elevation", "option,expected"], cases = [RiverCases, AllocationOptionCases])
def test_riv_allocation(active, top, bottom, stage, bottom_elevation, option, expected):
    actual = allocate_river_cells(option, active, top, bottom, stage, bottom_elevation)
    np.testing.assert_equal(actual.values[:, 0, 0] == expected)