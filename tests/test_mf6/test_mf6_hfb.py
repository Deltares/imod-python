import pathlib
import textwrap
import shapely.geometry as sg
import geopandas as gpd

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="module")
def domain():
    layer = [1, 2, 3]
    y = np.linspace(1.5, 0.5, num=9)
    x = np.linspace(0.5, 1.5, num=9)

    dy = y[1] - y[0]
    dx = x[1] - x[0]

    idomain = xr.DataArray(
        data=np.ones((3, 9, 9)),
        coords={"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy},
        dims=["layer", "y", "x"],
    )
    return idomain


@pytest.fixture(scope="module")
def line_gdf():
    line = sg.LineString([[1.25, 2.0], [0.75, 1.0]])
    line2 = sg.LineString([[1.25, 1.0], [0.75, 2.0]])

    return gpd.GeoDataFrame({"resistance": [100.0, 10.0]}, geometry=[line, line2])


@pytest.fixture(scope="module")
def hfb(domain, line_gdf):
    return imod.mf6.HorizontalFlowBarrier(idomain=domain, **line_gdf)


def test_render(hfb):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    actual = hfb.render(directory, "hfb", globaltimes)
    expected = textwrap.dedent(
        """\
        begin options

        end options

        begin dimensions
          maxhfb 48
        end dimensions

        begin period 1
          open/close mymodel/hfb/hfb.bin
        end period"""
    )

    assert expected == actual


def test_nmax_hfb(hfb):
    expected = 48
    actual = hfb._max_active_n()

    assert expected == actual
