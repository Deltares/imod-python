import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import (
    EvapotranspirationHighestActive,
    EvapotranspirationLayers,
    EvapotranspirationTopLayer,
)


@pytest.fixture(scope="function")
def evapotranspiration_top():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    maximum_rate = xr.DataArray(
        np.full((5, 5, 5), 1.0),
        coords={"time": datetimes, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("time", "y", "x"),
    )

    evt = EvapotranspirationTopLayer(
        maximum_rate=maximum_rate,
        surface=maximum_rate.copy(),
        extinction_depth=maximum_rate.copy(),
        save_budget=False,
    )
    return evt


@pytest.fixture(scope="function")
def evapotranspiration_layers():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    maximum_rate = xr.DataArray(
        np.full((5, 5, 5), 1.0),
        coords={"time": datetimes, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("time", "y", "x"),
    )

    evt = EvapotranspirationLayers(
        maximum_rate=maximum_rate,
        surface=maximum_rate.copy(),
        extinction_depth=maximum_rate.copy(),
        evapotranspiration_layer=maximum_rate.copy(),
        save_budget=False,
    )
    return evt


@pytest.fixture(scope="function")
def evapotranspiration_ha():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    maximum_rate = xr.DataArray(
        np.full((5, 5, 5), 1.0),
        coords={"time": datetimes, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("time", "y", "x"),
    )

    evt = EvapotranspirationHighestActive(
        maximum_rate=maximum_rate,
        surface=maximum_rate.copy(),
        extinction_depth=maximum_rate.copy(),
        save_budget=False,
    )
    return evt


def test_render__highest_top(evapotranspiration_top):
    evt = evapotranspiration_top
    directory = pathlib.Path(".")
    compare = textwrap.dedent(
        """\
        [evt]
            nevtop = 1
            ievtcb = 0
            evtr_p1 = maximum_rate_20000101000000.idf
            evtr_p2 = maximum_rate_20000102000000.idf
            evtr_p3 = maximum_rate_20000103000000.idf
            evtr_p4 = maximum_rate_20000104000000.idf
            evtr_p5 = maximum_rate_20000105000000.idf
            surf_p1 = surface_20000101000000.idf
            surf_p2 = surface_20000102000000.idf
            surf_p3 = surface_20000103000000.idf
            surf_p4 = surface_20000104000000.idf
            surf_p5 = surface_20000105000000.idf
            exdp_p1 = extinction_depth_20000101000000.idf
            exdp_p2 = extinction_depth_20000102000000.idf
            exdp_p3 = extinction_depth_20000103000000.idf
            exdp_p4 = extinction_depth_20000104000000.idf
            exdp_p5 = extinction_depth_20000105000000.idf"""
    )

    assert evt._render(directory, globaltimes=evt.time.values, nlayer=3) == compare


def test_render__layers(evapotranspiration_layers):
    evt = evapotranspiration_layers
    directory = pathlib.Path(".")
    compare = textwrap.dedent(
        """\
        [evt]
            nevtop = 2
            ievtcb = 0
            evtr_p1 = maximum_rate_20000101000000.idf
            evtr_p2 = maximum_rate_20000102000000.idf
            evtr_p3 = maximum_rate_20000103000000.idf
            evtr_p4 = maximum_rate_20000104000000.idf
            evtr_p5 = maximum_rate_20000105000000.idf
            surf_p1 = surface_20000101000000.idf
            surf_p2 = surface_20000102000000.idf
            surf_p3 = surface_20000103000000.idf
            surf_p4 = surface_20000104000000.idf
            surf_p5 = surface_20000105000000.idf
            exdp_p1 = extinction_depth_20000101000000.idf
            exdp_p2 = extinction_depth_20000102000000.idf
            exdp_p3 = extinction_depth_20000103000000.idf
            exdp_p4 = extinction_depth_20000104000000.idf
            exdp_p5 = extinction_depth_20000105000000.idf
            ievt_p1 = evapotranspiration_layer_20000101000000.idf
            ievt_p2 = evapotranspiration_layer_20000102000000.idf
            ievt_p3 = evapotranspiration_layer_20000103000000.idf
            ievt_p4 = evapotranspiration_layer_20000104000000.idf
            ievt_p5 = evapotranspiration_layer_20000105000000.idf"""
    )

    assert evt._render(directory, globaltimes=evt.time.values, nlayer=3) == compare


def test_render__highest_active(evapotranspiration_ha):
    evt = evapotranspiration_ha
    directory = pathlib.Path(".")
    compare = textwrap.dedent(
        """\
        [evt]
            nevtop = 3
            ievtcb = 0
            evtr_p1 = maximum_rate_20000101000000.idf
            evtr_p2 = maximum_rate_20000102000000.idf
            evtr_p3 = maximum_rate_20000103000000.idf
            evtr_p4 = maximum_rate_20000104000000.idf
            evtr_p5 = maximum_rate_20000105000000.idf
            surf_p1 = surface_20000101000000.idf
            surf_p2 = surface_20000102000000.idf
            surf_p3 = surface_20000103000000.idf
            surf_p4 = surface_20000104000000.idf
            surf_p5 = surface_20000105000000.idf
            exdp_p1 = extinction_depth_20000101000000.idf
            exdp_p2 = extinction_depth_20000102000000.idf
            exdp_p3 = extinction_depth_20000103000000.idf
            exdp_p4 = extinction_depth_20000104000000.idf
            exdp_p5 = extinction_depth_20000105000000.idf"""
    )

    assert evt._render(directory, globaltimes=evt.time.values, nlayer=3) == compare


@pytest.mark.parametrize("varname", ["maximum_rate", "surface", "extinction_depth"])
def test_render__stress_repeats(evapotranspiration_ha, varname):
    evt = evapotranspiration_ha.isel(time=0)
    directory = pathlib.Path(".")
    da = evt[varname]
    datetimes = pd.date_range("2000-01-01", "2000-01-03")
    da_transient = xr.concat(
        [da.assign_coords(time=t) for t in datetimes[:-1]], dim="time"
    )
    evt[varname] = da_transient

    stress_repeats = {datetimes[-1]: datetimes[0]}
    evt.repeat_stress(**{varname: stress_repeats})
    _ = evt._render(directory, globaltimes=datetimes, system_index=1, nlayer=3)
