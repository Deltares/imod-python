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
    dummy = ""  # Arguments go unused
    actual = hfb.render(dummy, dummy, dummy)
    expected = textwrap.dedent(
        """\
        begin options

        end options

        begin dimensions
          maxhfb 48
        end dimensions

        begin period 1
           1    2    4    1    2    5 1.000000e-02
           2    2    4    2    2    5 1.000000e-02
           3    2    4    3    2    5 1.000000e-02
           1    1    5    1    1    6 1.000000e-02
           2    1    5    2    1    6 1.000000e-02
           3    1    5    3    1    6 1.000000e-02
           1    1    5    1    2    5 1.000000e-02
           2    1    5    2    2    5 1.000000e-02
           3    1    5    3    2    5 1.000000e-02
           1    3    4    1    3    5 1.000000e-02
           2    3    4    2    3    5 1.000000e-02
           3    3    4    3    3    5 1.000000e-02
           1    3    4    1    4    4 1.000000e-02
           2    3    4    2    4    4 1.000000e-02
           3    3    4    3    4    4 1.000000e-02
           1    4    3    1    4    4 1.000000e-02
           2    4    3    2    4    4 1.000000e-02
           3    4    3    3    4    4 1.000000e-02
           1    5    3    1    5    4 1.000000e-02
           2    5    3    2    5    4 1.000000e-02
           3    5    3    3    5    4 1.000000e-02
           1    5    6    1    5    7 1.000000e-01
           2    5    6    2    5    7 1.000000e-01
           3    5    6    3    5    7 1.000000e-01
           1    4    6    1    4    7 1.000000e-01
           2    4    6    2    4    7 1.000000e-01
           3    4    6    3    4    7 1.000000e-01
           1    2    5    1    2    6 1.000000e-01
           2    2    5    2    2    6 1.000000e-01
           3    2    5    3    2    6 1.000000e-01
           1    3    5    1    3    6 1.000000e-01
           2    3    5    2    3    6 1.000000e-01
           3    3    5    3    3    6 1.000000e-01
           1    3    6    1    4    6 1.000000e-01
           2    3    6    2    4    6 1.000000e-01
           3    3    6    3    4    6 1.000000e-01
           1    2    5    1    2    6 1.000000e-01
           2    2    5    2    2    6 1.000000e-01
           3    2    5    3    2    6 1.000000e-01
           1    1    4    1    1    5 1.000000e-01
           2    1    4    2    1    5 1.000000e-01
           3    1    4    3    1    5 1.000000e-01
           1    1    5    1    2    5 1.000000e-01
           2    1    5    2    2    5 1.000000e-01
           3    1    5    3    2    5 1.000000e-01
           1    4    6    1    4    7 1.000000e-01
           2    4    6    2    4    7 1.000000e-01
           3    4    6    3    4    7 1.000000e-01
        end period"""
    )

    assert expected == actual


def test_nmax_hfb(hfb):
    expected = 48
    actual = hfb._max_active_n()

    assert expected == actual
