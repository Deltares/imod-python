import xarray as xr
from pytest_cases import parametrize_with_cases
from xarray.tests import raise_if_dask_computes

from imod.common.utilities.mask import _skip_dataarray
from imod.tests.fixtures.package_instance_creation import get_grid_da


class GridCases:
    def case_structured(self):
        return get_grid_da(False, float).chunk({"layer": 1})

    def case_unstructured(self):
        return get_grid_da(True, float).chunk({"layer": 1})


@parametrize_with_cases("grid", cases=GridCases)
def test_skip_dataarray(grid):
    layer_da = xr.DataArray([1, 2, 3], coords={"layer": [1, 2, 3]}, dims=("layer",))
    with raise_if_dask_computes():
        assert _skip_dataarray(grid) is False
        assert _skip_dataarray(xr.DataArray(True)) is True
        assert _skip_dataarray(layer_da) is True
