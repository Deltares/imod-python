import numpy as np
import pytest
import pyvista as pv
import xarray as xr

import imod
from imod import util


@pytest.fixture(scope="module")
def test_da():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol))
    da = xr.DataArray(data, **kwargs)
    return da


@pytest.fixture(scope="module")
def test_zda():
    nlay, nrow, ncol = 5, 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["z"] = np.arange(5.0) + 0.5
    kwargs = {"name": "z", "coords": coords, "dims": ("z", "y", "x")}
    data = np.ones((nlay, nrow, ncol), dtype=np.float32)
    da = xr.DataArray(data, **kwargs)
    return da


@pytest.fixture(scope="module")
def test_3dzda():
    nlay, nrow, ncol = 5, 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["layer"] = np.arange(nlay) + 1

    ncell = nlay * nrow * ncol
    scaled_addition = np.arange(1.0, 1.0 + ncell, 1.0) / ncell
    dz = (1.0 + scaled_addition).reshape((nlay, nrow, ncol))
    top = np.full((nlay, nrow, ncol), 0.0)
    top -= dz.cumsum(axis=0)
    bottom = top - dz

    dims = ("layer", "y", "x")
    coords["top"] = (dims, top)
    coords["bottom"] = (dims, bottom)
    data = np.ones((nlay, nrow, ncol))
    da = xr.DataArray(data, coords, dims)
    return da


def test_grid3_plane(tmp_path, test_da):
    da = test_da
    g = imod.visualize.grid_3d(da)
    assert g.bounds
    # g.plot(screenshot=tmp_path / "plane.png", off_screen=True)


def test_grid3d_z(tmp_path, test_zda):
    da = test_zda
    g = imod.visualize.grid_3d(da)
    assert g.bounds
    # g.plot(screenshot=tmp_path / "z.png", off_screen=True)


def test_grid3d_z3d(tmp_path, test_3dzda):
    da = test_3dzda
    g = imod.visualize.grid_3d(da)
    assert g.bounds
    # g.plot(screenshot=tmp_path / "z3d.png", off_screen=True)
