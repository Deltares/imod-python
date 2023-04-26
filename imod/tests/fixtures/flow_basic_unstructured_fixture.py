import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString

@pytest.fixture(scope="module")
def basic_unstructured_dis():
    """Basic model discretization"""

    shape = nlay, nrow, ncol = 3, 9, 9

    dx = 10.0
    dy = -10.0
    dz = np.array([5, 30, 100])
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    ibound = xr.DataArray(np.ones(shape, dtype=np.int32), coords=coords, dims=dims)

    surface = 0.0
    interfaces = np.insert((surface - np.cumsum(dz)), 0, surface)

    bottom = xr.DataArray(interfaces[1:], coords={"layer": layer}, dims="layer")
    top = xr.DataArray(interfaces[:-1], coords={"layer": layer}, dims="layer")

    return ibound, top, bottom
