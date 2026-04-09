import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xugrid as xu
from pytest_cases import parametrize_with_cases

import imod
from imod.typing.grid import full_like


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


def structured_grid():
    data = np.ones((3, 2, 2))
    coords = {"layer": [1, 2, 3], "y": [0.5, 1.5], "x": [0.5, 1.5]}
    dims = ("layer", "y", "x")

    return xr.DataArray(data, coords, dims)


class GridCases:
    def case_structured(self):
        return structured_grid()

    def case_unstructured(self):
        structured = structured_grid()
        return xu.UgridDataArray.from_structured(structured)


@parametrize_with_cases("da", cases=GridCases)
def test_convert_pointwaterhead_freshwaterhead_da(da):
    pwh = full_like(da, 4.0)
    dens = full_like(da, 1025.0)
    z = full_like(da, 1.0)
    fwh_expected = full_like(da, 4.075)
    fwh_actual = imod.evaluate.convert_pointwaterhead_freshwaterhead(pwh, dens, z)
    assert type(fwh_actual) is type(da)
    assert fwh_expected.equals(fwh_actual.round(5))


@parametrize_with_cases("da", cases=GridCases)
def test_convert_pointwaterhead_freshwaterhead_backfill(da):
    # edge case: point water head below z
    # return freshwater head of top underlying cell where elevation < pointwaterhead
    # only for grids with layers
    #
    # Test this by setting point water head in top layer to 0.0. This should
    # trigger a backfill with the value from layer 2, which is 4.075. So in the
    # end, test results should be 4.075 everywhere.
    pwh = full_like(da, 4.0)
    pwh[0, ...] = 0.0
    dens = full_like(da, 1025.0)
    z = full_like(da, 1.0)
    fwh_expected = full_like(da, 4.075)
    fwh_actual = imod.evaluate.convert_pointwaterhead_freshwaterhead(pwh, dens, z)
    assert type(fwh_actual) is type(da)
    assert fwh_expected.equals(fwh_actual.round(5))


@pytest.mark.parametrize("chunk", [False, True])
def test_calculate_gxg(chunk: bool):
    data = (np.ones((1, 2, 49)) * np.arange(49)).T
    coords = {
        "time": pd.date_range("2000-04-01", periods=49, freq="SMS"),
        "y": [0.5, 1.5],
        "x": [0.5],
    }
    dims = ("time", "y", "x")
    da = xr.DataArray(data, coords, dims)
    if chunk:
        da = da.chunk({"time": 1})

    coords_ref = {"y": [0.5, 1.5], "x": [0.5]}
    dims_ref = ("y", "x")
    glg_ref = (1 + 2 + 3 + 25 + 26 + 27) / 6
    ghg_ref = (22 + 23 + 24 + 46 + 47 + 48) / 6
    glg_ref = xr.DataArray(np.ones((2, 1)) * glg_ref, coords_ref, dims_ref)
    ghg_ref = xr.DataArray(np.ones((2, 1)) * ghg_ref, coords_ref, dims_ref)

    gxg = imod.evaluate.calculate_gxg(da, False).round(5)
    assert gxg["glg"].round(5).equals(glg_ref)
    assert gxg["ghg"].round(5).equals(ghg_ref)

    gxg = imod.evaluate.calculate_gxg(0 - da, True).round(5)
    assert gxg["glg"].round(5).equals(0 - glg_ref)
    assert gxg["ghg"].round(5).equals(0 - ghg_ref)


@pytest.mark.parametrize("chunk", [False, True])
def test_calculate_gxg_nan(chunk: bool):
    data = (np.ones((1, 2, 49)) * np.arange(49)).T
    # This invalidates the second year: it no longer forms a complete
    # "hydrological year" with 24 entries.
    data[:, 1, :] = np.nan
    data[-1, 0, :] = np.nan
    coords = {
        "time": pd.date_range("2000-04-01", periods=49, freq="SMS"),
        "y": [0.5, 1.5],
        "x": [0.5],
    }
    dims = ("time", "y", "x")
    da = xr.DataArray(data, coords, dims)
    if chunk:
        da = da.chunk({"time": 1})

    coords_ref = {"y": [0.5, 1.5], "x": [0.5]}
    dims_ref = ("y", "x")
    # Consequently, glg and ghg are based on only three values.
    glg_ref = np.array([[(1 + 2 + 3) / 3], [np.nan]])
    ghg_ref = np.array([[(22 + 23 + 24) / 3], [np.nan]])
    glg_ref = xr.DataArray(glg_ref, coords_ref, dims_ref)
    ghg_ref = xr.DataArray(ghg_ref, coords_ref, dims_ref)

    gxg = imod.evaluate.calculate_gxg(da, False).round(5)
    assert gxg["glg"].round(5).equals(glg_ref)
    assert gxg["ghg"].round(5).equals(ghg_ref)

    gxg = imod.evaluate.calculate_gxg(0 - da, True).round(5)
    assert gxg["glg"].round(5).equals(0 - glg_ref)
    assert gxg["ghg"].round(5).equals(0 - ghg_ref)


def test_calculate_gxg_points():
    df = pd.DataFrame()
    time = pd.date_range("2000-01-01", "2010-12-31")
    n_id = 10
    df["identification"] = np.repeat(np.arange(n_id), time.size).astype(str)
    df["piezometer head"] = np.tile(np.arange(time.size), n_id)
    df["datetime"] = np.tile(time, n_id)

    gxg = imod.evaluate.calculate_gxg_points(
        df,
        id="identification",
        time="datetime",
        head="piezometer head",
    )

    assert isinstance(gxg, pd.DataFrame)
    assert gxg.index.name == "identification"
    assert (gxg["n_years_gvg"] == 11).all()
    assert (gxg["n_years_gxg"] == 10).all()

    df.loc[df["identification"] == "0", "piezometer head"] = np.nan
    df.loc[
        (df["identification"] == "1") & (df["datetime"] < "2005-01-01"),
        "piezometer head",
    ] = np.nan

    gxg = imod.evaluate.calculate_gxg_points(
        df,
        id="identification",
        time="datetime",
        head="piezometer head",
    )
    assert gxg["n_years_gvg"].iloc[0] == 0
    assert gxg["n_years_gxg"].iloc[0] == 0
    assert np.isnan(gxg["glg"].iloc[0])
    assert np.isnan(gxg["gvg"].iloc[0])
    assert np.isnan(gxg["ghg"].iloc[0])
    assert gxg["n_years_gvg"].iloc[1] == 6
    assert gxg["n_years_gxg"].iloc[1] == 5
    assert np.isfinite(gxg["glg"].iloc[1])
    assert np.isfinite(gxg["gvg"].iloc[1])
    assert np.isfinite(gxg["ghg"].iloc[1])
