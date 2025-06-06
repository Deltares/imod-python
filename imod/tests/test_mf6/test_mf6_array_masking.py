import numpy as np
import pytest_cases
import xarray as xr
import xugrid as xu

from imod.common.utilities.mask import mask_arrays


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


def case_structured_arrays() -> tuple[xr.DataArray, xr.DataArray]:
    return create_structured_arrays()


def case_unstructured_arrays() -> tuple[xu.UgridDataArray, xu.UgridDataArray]:
    array1, array2 = create_structured_arrays()

    # Convert to unstructured arrays
    ugrid1 = xu.UgridDataArray.from_structured2d(array1)
    ugrid2 = xu.UgridDataArray.from_structured2d(array2)

    return ugrid1, ugrid2


@pytest_cases.parametrize_with_cases(
    "arrays",
    cases=".",
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
