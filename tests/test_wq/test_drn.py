import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import Drainage


@pytest.fixture(scope="module")
def drainage(request):
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    elevation = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    conductance = elevation.copy()

    drn = Drainage(elevation=elevation, conductance=conductance)
    return drn


def test_render(drainage):
    drn = drainage
    directory = pathlib.Path(".")

    compare = (
        "\n"
        "    elevation_p?_s1_l1 = elevation_l1.idf\n"
        "    elevation_p?_s1_l2 = elevation_l2.idf\n"
        "    elevation_p?_s1_l3 = elevation_l3.idf\n"
        "    cond_p?_s1_l1 = conductance_l1.idf\n"
        "    cond_p?_s1_l2 = conductance_l2.idf\n"
        "    cond_p?_s1_l3 = conductance_l3.idf"
    )

    assert drn._render(directory, globaltimes=["?"], system_index=1) == compare


def test_render_with_time(drainage):
    drn = drainage.copy()
    directory = pathlib.Path(".")
    elev = drn["elevation"]
    datetimes = pd.date_range("2000-01-01", "2000-01-02")

    elev_transient = xr.concat(
        [elev.assign_coords(time=t) for t in datetimes], dim="time"
    )
    drn["elevation"] = elev_transient

    compare = (
        "\n"
        "    elevation_p1_s1_l1 = elevation_20000101000000_l1.idf\n"
        "    elevation_p1_s1_l2 = elevation_20000101000000_l2.idf\n"
        "    elevation_p1_s1_l3 = elevation_20000101000000_l3.idf\n"
        "    elevation_p2_s1_l1 = elevation_20000102000000_l1.idf\n"
        "    elevation_p2_s1_l2 = elevation_20000102000000_l2.idf\n"
        "    elevation_p2_s1_l3 = elevation_20000102000000_l3.idf\n"
        "    cond_p?_s1_l1 = conductance_l1.idf\n"
        "    cond_p?_s1_l2 = conductance_l2.idf\n"
        "    cond_p?_s1_l3 = conductance_l3.idf"
    )

    assert drn._render(directory, globaltimes=datetimes, system_index=1) == compare


def test_render_with_timemap(drainage):
    drn = drainage
    directory = pathlib.Path(".")
    elev = drn["elevation"]
    datetimes = pd.date_range("2000-01-01", "2000-01-03")

    elev_transient = xr.concat(
        [elev.assign_coords(time=t) for t in datetimes[:-1]], dim="time"
    )
    drn["elevation"] = elev_transient
    timemap = {datetimes[-1]: datetimes[0]}
    drn.add_timemap(elevation=timemap)

    compare = (
        "\n"
        "    elevation_p1_s1_l1 = elevation_20000101000000_l1.idf\n"
        "    elevation_p1_s1_l2 = elevation_20000101000000_l2.idf\n"
        "    elevation_p1_s1_l3 = elevation_20000101000000_l3.idf\n"
        "    elevation_p2_s1_l1 = elevation_20000102000000_l1.idf\n"
        "    elevation_p2_s1_l2 = elevation_20000102000000_l2.idf\n"
        "    elevation_p2_s1_l3 = elevation_20000102000000_l3.idf\n"
        "    elevation_p3_s1_l1 = elevation_20000101000000_l1.idf\n"
        "    elevation_p3_s1_l2 = elevation_20000101000000_l2.idf\n"
        "    elevation_p3_s1_l3 = elevation_20000101000000_l3.idf\n"
        "    cond_p?_s1_l1 = conductance_l1.idf\n"
        "    cond_p?_s1_l2 = conductance_l2.idf\n"
        "    cond_p?_s1_l3 = conductance_l3.idf"
    )

    assert drn._render(directory, globaltimes=datetimes, system_index=1) == compare
