import dask
import numpy as np
import pytest
import xarray as xr

import imod


def test_where():
    a = xr.DataArray(
        [[0.0, 1.0], [2.0, np.nan]],
        {"y": [1.5, 0.5], "x": [0.5, 1.5]},
        ["y", "x"],
    )
    cond = a <= 1
    actual = imod.util.structured.where(cond, if_true=a, if_false=1.0)
    assert np.allclose(actual.values, [[0.0, 1.0], [1.0, np.nan]], equal_nan=True)

    actual = imod.util.structured.where(cond, if_true=0.0, if_false=1.0)
    assert np.allclose(actual.values, [[0.0, 0.0], [1.0, 1.0]], equal_nan=True)

    actual = imod.util.structured.where(cond, if_true=a, if_false=1.0, keep_nan=False)
    assert np.allclose(actual.values, [[0.0, 1.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="at least one of"):
        imod.util.structured.where(False, 1, 0)


def test_replace():
    # replace scalar
    da = xr.DataArray([0, 1, 2])
    out = imod.util.structured.replace(da, 1, 10)
    assert out.equals(xr.DataArray([0, 10, 2]))

    # Replace NaN by scalar
    da = xr.DataArray([np.nan, 1.0, 2.0])
    out = imod.util.structured.replace(da, np.nan, 10.0)
    assert out.equals(xr.DataArray([10.0, 1.0, 2.0]))

    # replace two
    da = xr.DataArray([0, 1, 2])
    out = imod.util.structured.replace(da, [1, 2], [10, 20])
    assert out.equals(xr.DataArray([0, 10, 20]))

    # With a NaN in the data
    da = xr.DataArray([np.nan, 1.0, 2.0])
    out = imod.util.structured.replace(da, [1, 2], [10, 20])
    assert out.equals(xr.DataArray([np.nan, 10.0, 20.0]))

    # Replace a NaN value
    da = xr.DataArray([np.nan, 1.0, 2.0])
    out = imod.util.structured.replace(da, [np.nan, 2], [10, 20])
    assert out.equals(xr.DataArray([10.0, 1.0, 20.0]))

    # With non-present values in to_replace
    da = xr.DataArray([np.nan, 1.0, 1.0, 2.0])
    out = imod.util.structured.replace(da, [1.0, 2.0, 30.0], [10.0, 20.0, 30.0])
    assert out.equals(xr.DataArray([np.nan, 10.0, 10.0, 20.0]))

    # With a nan and non-present values
    da = xr.DataArray([np.nan, 1.0, 1.0, 2.0])
    out = imod.util.structured.replace(da, [np.nan, 1.0, 2.0, 30.0], 10.0)
    assert out.equals(xr.DataArray([10.0, 10.0, 10.0, 10.0]))

    # With a dask array
    da = xr.DataArray(dask.array.full(3, 1.0))
    out = imod.util.structured.replace(da, [1.0, 2.0], [10.0, 20.0])
    assert isinstance(out.data, dask.array.Array)
    assert out.equals(xr.DataArray([10.0, 10.0, 10.0]))

    # scalar to_replace, non-scalar value
    with pytest.raises(TypeError):
        imod.util.structured.replace(da, 1.0, [10.0, 20.0])

    # 2D arrays
    with pytest.raises(ValueError):
        imod.util.structured.replace(da, [[1.0, 2.0]], [[10.0, 20.0]])

    # 1D to_replace, 2D value
    with pytest.raises(ValueError):
        imod.util.structured.replace(da, [1.0, 2.0], [[10.0, 20.0]])

    # 1D, different size
    with pytest.raises(ValueError):
        imod.util.structured.replace(da, [1.0, 2.0], [10.0, 20.0, 30.0])
