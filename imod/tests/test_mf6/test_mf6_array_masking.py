import numpy as np
import xarray as xr

from imod.mf6.utilities.mask import mask_arrays


def test_array_masking():
    x = [1]
    y = [1, 2, 3]
    layer = [1, 2]
    coords = {"layer": layer, "y": y, "x": x}
    dims = ("layer", "y", "x")

    array1 = xr.DataArray([[[1], [1], [1]], [[1], [1], [1]]], coords=coords, dims=dims)
    array2 = xr.DataArray(
        [[[np.nan], [1], [1]], [[1], [1], [1]]], coords=coords, dims=dims
    )

    masked_arrays = mask_arrays({"array1": array1, "array2": array2})

    # element 0,0,0 should be nan in both arrays
    assert np.isnan(masked_arrays["array1"].values[0, 0, 0])
    assert np.isnan(masked_arrays["array2"].values[0, 0, 0])

    # there should be only 1 nan in both arrays
    masked_arrays["array1"].values[0, 0, 0] = 1
    masked_arrays["array2"].values[0, 0, 0] = 1
    assert np.all(~np.isnan(masked_arrays["array1"].values))
    assert np.all(~np.isnan(masked_arrays["array2"].values))
