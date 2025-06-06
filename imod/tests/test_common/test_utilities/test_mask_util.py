import numpy as np
import pytest
import xarray as xr
import xugrid as xu
from pytest_cases import parametrize_with_cases
from xarray.tests import raise_if_dask_computes

from imod.common.utilities.mask import (
    _skip_dataarray,
    broadcast_and_mask_arrays,
    mask_arrays,
)
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


def create_structured_arrays() -> tuple[xr.DataArray, xr.DataArray]:
    x = [1]
    y = [1, 2, 3]
    layer = [1, 2]
    coords = {"layer": layer, "y": y, "x": x, "dx": 1, "dy": 1}
    dims = ("layer", "y", "x")

    array1 = xr.DataArray([[[1], [1], [1]], [[1], [1], [1]]], coords=coords, dims=dims)
    array2 = xr.DataArray(
        [[[np.nan], [1], [1]], [[1], [1], [1]]], coords=coords, dims=dims
    )

    return array1, array2


class MaskArrayCases:
    def case_structured(self):
        return create_structured_arrays()

    def case_unstructured(self):
        array1, array2 = create_structured_arrays()

        # Convert to unstructured arrays
        ugrid1 = xu.UgridDataArray.from_structured2d(array1)
        ugrid2 = xu.UgridDataArray.from_structured2d(array2)

        return ugrid1, ugrid2


@parametrize_with_cases(
    "arrays",
    cases=MaskArrayCases,
)
def test_array_masking(arrays):
    array1, array2 = arrays

    masked_arrays = mask_arrays({"array1": array1, "array2": array2})

    # element first element should be nan in both arrays
    array1_1d = masked_arrays["array1"].values.ravel()
    array2_1d = masked_arrays["array2"].values.ravel()
    assert np.isnan(array1_1d[0])
    assert np.isnan(array2_1d[0])

    # there should be only 1 nan in both arrays
    array1_1d[0] = 1
    array2_1d[0] = 1
    assert np.all(~np.isnan(array1_1d))
    assert np.all(~np.isnan(array2_1d))


def test_broadcast_and_mask_arrays():
    array1, array2 = create_structured_arrays()
    layer = [1, 2]
    scalar_array = xr.DataArray([1.0, 1.0], coords={"layer": layer}, dims=("layer",))
    # Test broadcasting and masking with two arrays with the same shape
    result1 = broadcast_and_mask_arrays({"array1": array1, "array2": array2})
    xr.testing.assert_equal(
        result1["array1"], array2
    )  # Masking turns array1 into array2
    xr.testing.assert_equal(result1["array2"], array2)
    # Test broadcasting and masking with one array and a scalar array
    result2 = broadcast_and_mask_arrays(
        {"array1": array1, "not_a_scalar_array": scalar_array}
    )
    xr.testing.assert_equal(result2["array1"], array1)
    xr.testing.assert_identical(result2["not_a_scalar_array"], array1)
    # No grid should result in a ValueError
    with pytest.raises(ValueError):
        broadcast_and_mask_arrays({"array1": scalar_array, "array2": scalar_array})
