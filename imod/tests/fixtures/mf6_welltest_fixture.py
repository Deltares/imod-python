import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def mf6wel_test_data_stationary():
    cellid_values = np.array(
        [
            [1, 1, 9],
            [1, 2, 9],
            [1, 1, 8],
            [1, 2, 8],
            [2, 3, 7],
            [2, 4, 7],
            [2, 3, 6],
            [2, 4, 6],
        ],
    )
    coords = {"ncellid": np.arange(8) + 1, "nmax_cellid": ["layer", "row", "column"]}
    cellid = xr.DataArray(cellid_values, coords=coords, dims=("ncellid", "nmax_cellid"))
    rate = xr.DataArray(
        [1.0] * 8, coords={"ncellid": np.arange(8) + 1}, dims=("ncellid",)
    )
    return cellid, rate


@pytest.fixture(scope="session")
def mf6wel_test_data_transient():
    cellid_values = np.array(
        [
            [1, 1, 9],
            [1, 2, 9],
            [1, 1, 8],
            [1, 2, 8],
            [2, 3, 7],
            [2, 4, 7],
            [2, 3, 6],
            [2, 4, 6],
        ],
    )
    coords = {"ncellid": np.arange(8) + 1, "nmax_cellid": ["layer", "row", "column"]}
    cellid = xr.DataArray(cellid_values, coords=coords, dims=("ncellid", "nmax_cellid"))

    rate = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    globaltimes = pd.date_range("2000-01-01", "2000-01-06")
    weltimes = globaltimes[:-1]

    rate_time = xr.DataArray(
        np.arange(len(weltimes)) + 1, coords={"time": weltimes}, dims=("time",)
    )
    rate_cells = xr.DataArray(
        rate, coords={"ncellid": np.arange(8) + 1}, dims=("ncellid",)
    )
    rate_wel = rate_time * rate_cells

    return cellid, rate_wel


@pytest.fixture(scope="session")
def well_high_lvl_test_data_stationary():
    screen_top = [0.0, 0.0, 0.0, 0.0, -6.0, -6.0, -6.0, -6.0]
    screen_bottom = [-2.0, -2.0, -2.0, -2.0, -20.0, -20.0, -20.0, -20.0]
    # fmt: off
    y = [83.0, 77.0, 82.0, 71.0, 62.0, 52.0, 66.0, 59.0]
    x = [81.0, 82.0, 75.0, 77.0, 68.0, 64.0, 52.0, 51.0]
    # fmt: on
    rate = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    rate_wel = xr.DataArray(rate, dims=("index",))

    da_species = xr.DataArray(
        [10, 23],
        coords={"species": ["salinity", "temparature"]},
        dims=("species",),
    )

    concentration = da_species * rate_wel

    return screen_top, screen_bottom, y, x, rate_wel, concentration


@pytest.fixture(scope="session")
def well_high_lvl_test_data_transient():
    screen_top = [0.0, 0.0, 0.0, 0.0, -6.0, -6.0, -6.0, -6.0]
    screen_bottom = [-2.0, -2.0, -2.0, -2.0, -20.0, -20.0, -20.0, -20.0]
    # fmt: off
    y = [83.0, 77.0, 82.0, 71.0, 62.0, 52.0, 66.0, 59.0]
    x = [81.0, 82.0, 75.0, 77.0, 68.0, 64.0, 52.0, 51.0]
    # fmt: on
    rate = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    globaltimes = pd.date_range("2000-01-01", "2000-01-06")
    weltimes = globaltimes[:-1]

    rate_time = xr.DataArray(
        np.arange(len(weltimes)) + 1, coords={"time": weltimes}, dims=("time",)
    )
    rate_cells = xr.DataArray(rate, dims=("index",))
    rate_wel = rate_time * rate_cells

    da_species = xr.DataArray(
        [10, 23],
        coords={"species": ["salinity", "temparature"]},
        dims=("species",),
    )

    concentration = da_species * rate_wel

    return screen_top, screen_bottom, y, x, rate_wel, concentration


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
