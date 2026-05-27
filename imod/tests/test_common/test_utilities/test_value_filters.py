import xarray as xr
import dask
import numpy as np
import pytest

from imod.common.utilities.value_filters import enforce_scalar, is_empty_dataarray

def test_enforce_scalar():
    da = xr.DataArray([42])
    assert enforce_scalar(da) == 42

    da = xr.DataArray([42.0])
    assert enforce_scalar(da) == 42.0

    da = xr.DataArray([False])
    assert enforce_scalar(da) is False

    da = xr.DataArray([None])
    assert enforce_scalar(da) is None

    da = xr.DataArray(["test"])
    assert enforce_scalar(da) == "test"

    data = dask.array.from_array([False], chunks=1)
    da = xr.DataArray(data)
    assert enforce_scalar(da) == False

    with pytest.raises(ValueError):
        da = xr.DataArray([1, 2])
        enforce_scalar(da)

def test_is_empty_dataarray():
    da = xr.DataArray([1, 2, 3])
    assert not is_empty_dataarray(da)

    da = xr.DataArray([None, None, None])
    assert is_empty_dataarray(da)

    da = xr.DataArray([np.nan, np.nan])
    assert is_empty_dataarray(da)

    da = xr.DataArray([1, None, 3])
    assert not is_empty_dataarray(da)

    da = xr.DataArray([1, np.nan, 3])
    assert not is_empty_dataarray(da)

    da = "not a DataArray"
    assert not is_empty_dataarray(da)

    data = dask.array.from_array([np.nan, np.nan], chunks=2)
    da = xr.DataArray(data)
    assert is_empty_dataarray(da)

    #TODO: Figure out if this edge case needs to be handled or not.
    data = dask.array.from_array([None, None], chunks=2)
    da = xr.DataArray(data)
    assert is_empty_dataarray(da)