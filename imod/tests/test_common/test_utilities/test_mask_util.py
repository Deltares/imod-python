import xarray as xr
from pytest_cases import parametrize_with_cases
from xarray.tests import raise_if_dask_computes

from imod.common.utilities.mask import _skip_dataarray
from imod.tests.fixtures.package_instance_creation import get_grid_da


class DataArrayCases:
    def case_structured(self):
        grid = get_grid_da(False, float).chunk({"layer": 1})
        return grid, False

    def case_unstructured(self):
        grid = get_grid_da(True, float).chunk({"layer": 1})
        return grid, False

    def case_layered_constant(self):
        layered_constant = xr.DataArray(
            [1, 2, 3], coords={"layer": [1, 2, 3]}, dims=("layer",)
        ).chunk({"layer": 1})
        return layered_constant, True

    def case_constant(self):
        return xr.DataArray(True).chunk({}), True


@parametrize_with_cases("da, expected", cases=DataArrayCases)
def test_skip_dataarray(da, expected):
    with raise_if_dask_computes():
        assert _skip_dataarray(da) is expected
