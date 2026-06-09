import dask
import numpy as np
import pytest
import pytest_cases
import xarray as xr

from imod.common.utilities.value_filters import enforce_scalar, is_empty_dataarray


class ScalarCases:
    def case_int(self):
        return 42

    def case_float(self):
        return 42.0

    def case_bool(self):
        return False

    def case_none(self):
        return None

    def case_string(self):
        return "test"


@pytest_cases.parametrize_with_cases("input_value", cases=ScalarCases)
def test_enforce_scalar(input_value):
    da = xr.DataArray([input_value])
    assert enforce_scalar(da) == input_value

    da = xr.DataArray([input_value, input_value])
    with pytest.raises(ValueError):
        enforce_scalar(da)


@pytest_cases.parametrize_with_cases("input_value", cases=ScalarCases)
def test_enforce_scalar_from_dask(input_value):
    data = dask.array.from_array([input_value], chunks=1)
    da = xr.DataArray(data)
    assert enforce_scalar(da) == input_value

    data = dask.array.from_array([input_value, input_value], chunks=2)
    da = xr.DataArray(data)
    with pytest.raises(ValueError):
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
