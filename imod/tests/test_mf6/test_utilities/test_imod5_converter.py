import pytest
import xarray as xr
from pytest_cases import parametrize_with_cases

from imod.mf6.utilities.imod5_converter import convert_ibound_to_idomain
from imod.util import empty_3d


@pytest.fixture(scope="function")
def template_grid():
    dx = 50.0
    dy = -50.0
    xmin = 0.0
    xmax = 100.0
    ymin = 0.0
    ymax = 100.0
    layer = [1, 2, 3, 4, 5]

    return empty_3d(dx, xmin, xmax, dy, ymin, ymax, layer).fillna(1.0)


class IboundCases:
    def case_active(self):
        thickness = [1.0, 1.0, 1.0, 1.0, 1.0]
        ibound = [1, 1, 1, 1, 1]
        idomain = [1, 1, 1, 1, 1]
        return thickness, ibound, idomain

    def case_inactive(self):
        thickness = [1.0, 1.0, 1.0, 1.0, 1.0]
        ibound = [0, 1, 1, 1, 0]
        idomain = [0, 1, 1, 1, 0]
        return thickness, ibound, idomain

    def case_min1(self):
        thickness = [1.0, 1.0, 1.0, 1.0, 1.0]
        ibound = [1, -1, 1, -1, 0]
        idomain = [1, 1, 1, 1, 0]
        return thickness, ibound, idomain

    def case_all_inactive(self):
        thickness = [1.0, 0.0, 0.0, 0.0, 1.0]
        ibound = [0, 0, 0, 0, 0]
        idomain = [0, 0, 0, 0, 0]
        return thickness, ibound, idomain

    def case_vpt(self):
        thickness = [1.0, 0.0, 1.0, 0.0, 1.0]
        ibound = [1, 1, 1, 1, 1]
        idomain = [1, -1, 1, -1, 1]
        return thickness, ibound, idomain

    def case_vpt_zero_thickness_at_edge(self):
        thickness = [1.0, 0.0, 1.0, 0.0, 1.0]
        ibound = [0, 1, 1, 1, 0]
        idomain = [0, 0, 1, 0, 0]
        return thickness, ibound, idomain

    def case_mixed(self):
        thickness = [1.0, 0.0, 1.0, 0.0, 1.0]
        ibound = [1, -1, 1, 1, 0]
        idomain = [1, -1, 1, 0, 0]
        return thickness, ibound, idomain


@parametrize_with_cases(argnames="thickness,ibound,expected", cases=IboundCases)
def test_convert_ibound_to_idomain(template_grid, thickness, ibound, expected):
    layer = template_grid.coords["layer"]
    thickness = xr.ones_like(layer) * thickness * template_grid
    ibound = xr.ones_like(layer) * ibound * template_grid
    expected = xr.ones_like(layer) * expected * template_grid

    actual = convert_ibound_to_idomain(ibound, thickness)

    assert actual.equals(expected)
