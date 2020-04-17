import numpy as np
import pandas as pd
import xarray as xr

import imod


def test_convert_pointwaterhead_freshwaterhead_scalar():
    # fresh water
    assert (
        round(imod.evaluate.convert_pointwaterhead_freshwaterhead(4.0, 1000.0, 1.0), 5)
        == 4.0
    )

    # saline
    assert (
        round(imod.evaluate.convert_pointwaterhead_freshwaterhead(4.0, 1025.0, 1.0), 5)
        == 4.075
    )


def test_convert_pointwaterhead_freshwaterhead_da():
    data = np.ones((3, 2, 1))
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5]}
    dims = ("layer", "y", "x")

    da = xr.DataArray(data, coords, dims)
    pwh = xr.full_like(da, 4.0)
    dens = xr.full_like(da, 1025.0)
    z = xr.full_like(da, 1.0)
    fwh = xr.full_like(da, 4.075)
    fwh2 = imod.evaluate.convert_pointwaterhead_freshwaterhead(pwh, dens, z)
    assert fwh.equals(fwh2.round(5))


def test_calculate_gxg():
    data = (np.ones((1, 2, 49)) * np.arange(49)).T
    coords = {
        "time": pd.date_range("2000-04-01", periods=49, freq="SMS"),
        "y": [0.5, 1.5],
        "x": [0.5],
    }
    dims = ("time", "y", "x")
    da = xr.DataArray(data, coords, dims)

    coords_ref = {"y": [0.5, 1.5], "x": [0.5]}
    dims_ref = ("y", "x")
    glg_ref = (1 + 2 + 3 + 25 + 26 + 27) / 6
    ghg_ref = (22 + 23 + 24 + 46 + 47 + 48) / 6
    attrs = {"GxG_period": "2000-04-01 - 2002-04-01"}
    glg_ref = xr.DataArray(np.ones((2, 1)) * glg_ref, coords_ref, dims_ref, attrs=attrs)
    ghg_ref = xr.DataArray(np.ones((2, 1)) * ghg_ref, coords_ref, dims_ref, attrs=attrs)

    gxg = imod.evaluate.calculate_gxg(da, False).round(5)
    assert gxg["glg"].round(5).equals(glg_ref)
    assert gxg["ghg"].round(5).equals(ghg_ref)

    gxg = imod.evaluate.calculate_gxg(0 - da, True).round(5)
    assert gxg["glg"].round(5).equals(0 - glg_ref)
    assert gxg["ghg"].round(5).equals(0 - ghg_ref)


def test_calculate_gxg_nan():
    data = (np.ones((1, 2, 49)) * np.arange(49)).T
    data[:, 1, :] = np.nan
    data[-1, 0, :] = np.nan
    coords = {
        "time": pd.date_range("2000-04-01", periods=49, freq="SMS"),
        "y": [0.5, 1.5],
        "x": [0.5],
    }
    dims = ("time", "y", "x")
    da = xr.DataArray(data, coords, dims)

    coords_ref = {"y": [0.5, 1.5], "x": [0.5]}
    dims_ref = ("y", "x")
    glg_ref = np.array([[(1 + 2 + 3 + 25 + 26 + 27) / 6], [np.nan]])
    ghg_ref = np.array([[(22 + 23 + 24 + 45 + 46 + 47) / 6], [np.nan]])
    attrs = {"GxG_period": "2000-04-01 - 2002-04-01"}
    glg_ref = xr.DataArray(glg_ref, coords_ref, dims_ref, attrs=attrs)
    ghg_ref = xr.DataArray(ghg_ref, coords_ref, dims_ref, attrs=attrs)

    gxg = imod.evaluate.calculate_gxg(da, False).round(5)
    assert gxg["glg"].round(5).equals(glg_ref)
    assert gxg["ghg"].round(5).equals(ghg_ref)

    gxg = imod.evaluate.calculate_gxg(0 - da, True).round(5)
    assert gxg["glg"].round(5).equals(0 - glg_ref)
    assert gxg["ghg"].round(5).equals(0 - ghg_ref)
