import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def well_test_data_stationary():
    layer = np.array([3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    row = np.array([5, 4, 6, 9, 9, 9, 9, 11, 11, 11, 11, 13, 13, 13, 13])
    column = np.array([11, 6, 12, 8, 10, 12, 14, 8, 10, 12, 14, 8, 10, 12, 14])
    rate = np.full(15, 5.0)
    injection_concentration = np.full((15, 2), np.NaN)
    injection_concentration[:, 0] = 123  # salinity
    injection_concentration[:, 1] = 456  # temperature
    return layer, row, column, rate, injection_concentration


@pytest.fixture(scope="session")
def well_test_data_transient():
    layer = np.array([3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    row = np.array([5, 4, 6, 9, 9, 9, 9, 11, 11, 11, 11, 13, 13, 13, 13])
    column = np.array([11, 6, 12, 8, 10, 12, 14, 8, 10, 12, 14, 8, 10, 12, 14])
    times = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]")
    rate = xr.DataArray(
        np.full((2, 15), 5.0), coords={"time": times}, dims=["time", "index"]
    )
    injection_concentration = np.full((2, 15, 2), np.NaN)
    injection_concentration[0, :, 0] = 123  # salinity, time 0
    injection_concentration[0, :, 1] = 456  # temperature, time 0
    injection_concentration[1, :, 0] = 246  # salinity, time 1
    injection_concentration[1, :, 1] = 135  # temperature, time 1
    return layer, row, column, times, rate, injection_concentration
