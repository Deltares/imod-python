import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import RechargeHighestActive, RechargeLayers, RechargeTopLayer


@pytest.fixture(scope="function")
def recharge_top():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    rate = xr.DataArray(
        np.full((5, 5, 5), 1.0),
        coords={"time": datetimes, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("time", "y", "x"),
    )

    rch = RechargeTopLayer(rate=rate, concentration=rate.copy(), save_budget=False)
    return rch


@pytest.fixture(scope="function")
def recharge_layers():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    rate = xr.DataArray(
        np.full((5, 5, 5), 1.0),
        coords={"time": datetimes, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("time", "y", "x"),
    )

    rch = RechargeLayers(
        rate=rate,
        recharge_layer=rate.copy(),
        concentration=rate.copy(),
        save_budget=False,
    )
    return rch


@pytest.fixture(scope="function")
def recharge_ha():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    rate = xr.DataArray(
        np.full((5, 5, 5), 1.0),
        coords={"time": datetimes, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("time", "y", "x"),
    )

    rch = RechargeHighestActive(rate=rate, concentration=rate.copy(), save_budget=False)
    return rch


def test_render__highest_top(recharge_top):
    rch = recharge_top
    directory = pathlib.Path(".")
    compare = textwrap.dedent(
        """\
        [rch]
            nrchop = 1
            irchcb = 0
            rech_p1 = rate_20000101000000.idf
            rech_p2 = rate_20000102000000.idf
            rech_p3 = rate_20000103000000.idf
            rech_p4 = rate_20000104000000.idf
            rech_p5 = rate_20000105000000.idf"""
    )

    assert rch._render(directory, globaltimes=rch.time.values) == compare


def test_render__layers(recharge_layers):
    rch = recharge_layers
    directory = pathlib.Path(".")
    compare = textwrap.dedent(
        """\
        [rch]
            nrchop = 2
            irchcb = 0
            rech_p1 = rate_20000101000000.idf
            rech_p2 = rate_20000102000000.idf
            rech_p3 = rate_20000103000000.idf
            rech_p4 = rate_20000104000000.idf
            rech_p5 = rate_20000105000000.idf
            irch_p1 = recharge_layer_20000101000000.idf
            irch_p2 = recharge_layer_20000102000000.idf
            irch_p3 = recharge_layer_20000103000000.idf
            irch_p4 = recharge_layer_20000104000000.idf
            irch_p5 = recharge_layer_20000105000000.idf"""
    )

    assert rch._render(directory, globaltimes=rch.time.values) == compare


def test_render__highest_active(recharge_ha):
    rch = recharge_ha
    directory = pathlib.Path(".")
    compare = textwrap.dedent(
        """\
        [rch]
            nrchop = 3
            irchcb = 0
            rech_p1 = rate_20000101000000.idf
            rech_p2 = rate_20000102000000.idf
            rech_p3 = rate_20000103000000.idf
            rech_p4 = rate_20000104000000.idf
            rech_p5 = rate_20000105000000.idf"""
    )

    assert rch._render(directory, globaltimes=rch.time.values) == compare


def test_ssm_cellcount_scalar_highest_active(recharge_ha):
    rch_template = recharge_ha
    rate = rch_template["rate"]
    rate[:, 0, :] = np.nan
    rch = RechargeHighestActive(rate=rate, concentration=0.1)
    ibound = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={
            "layer": [1, 2, 3],
            "y": recharge_ha["y"],
            "x": recharge_ha["x"],
            "dx": 1.0,
            "dy": -1.0,
        },
        dims=("layer", "y", "x"),
    )
    ibound[0, 0, :] = 0.0

    rch._set_ssm_layers(ibound)
    assert np.allclose(rch._ssm_layers, [1])


@pytest.mark.parametrize("varname", ["rate", "concentration"])
def test_render__timemap(recharge_ha, varname):
    rch = recharge_ha.isel(time=0)
    directory = pathlib.Path(".")
    da = rch[varname]
    datetimes = pd.date_range("2000-01-01", "2000-01-03")
    da_transient = xr.concat(
        [da.assign_coords(time=t) for t in datetimes[:-1]], dim="time"
    )
    rch[varname] = da_transient

    timemap = {datetimes[-1]: datetimes[0]}
    rch.add_timemap(**{varname: timemap})
    actual = rch._render(directory, globaltimes=datetimes, system_index=1)
