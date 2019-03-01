from glob import glob
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import cftime
from imod import idf


@pytest.fixture(scope="module")
def test_da(request):
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = idf._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)

    def remove():
        try:
            os.remove("test.idf")
        except FileNotFoundError:
            pass

    request.addfinalizer(remove)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_layerda(request):
    nlay, nrow, ncol = 5, 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = idf._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["layer"] = np.arange(nlay) + 1
    kwargs = {"name": "layer", "coords": coords, "dims": ("layer", "y", "x")}
    data = np.ones((nlay, nrow, ncol), dtype=np.float32)

    def remove():
        paths = glob("layer_l[0-9].idf")
        for p in paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    request.addfinalizer(remove)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_da_nonequidistant(request):
    nrow, ncol = 3, 4
    dx = np.array([0.9, 1.1, 0.8, 1.2])
    dy = np.array([-1.3, -0.7, -1.0])
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = idf._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "nonequidistant", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)

    def remove():
        try:
            os.remove("nonequidistant.idf")
        except FileNotFoundError:
            pass

    request.addfinalizer(remove)
    return xr.DataArray(data, **kwargs)


def test_xycoords_equidistant():
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = idf._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    assert np.allclose(coords["x"], np.arange(xmin + dx / 2.0, xmax, dx))
    assert np.allclose(coords["y"], np.arange(ymax + dy / 2.0, ymin, dy))
    assert coords["dx"] == dx
    assert coords["dy"] == dy


def test_xycoords_nonequidistant():
    dx = np.array([0.9, 1.1, 0.8, 1.2])
    dy = np.array([-1.3, -0.7, -1.0])
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = idf._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    assert np.allclose(coords["x"], np.array([0.45, 1.45, 2.4, 3.4]))
    assert np.allclose(coords["y"], np.array([2.35, 1.35, 0.5]))
    assert coords["dx"][0] == "x"
    assert np.allclose(coords["dx"][1], dx)
    assert coords["dy"][0] == "y"
    assert np.allclose(coords["dy"][1], dy)


def test_saveload(test_da):
    idf.save("test.idf", test_da)
    assert Path("test.idf").exists()
    da = idf.load("test.idf")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_da)
    with pytest.warns(FutureWarning):
        idf.load("test.idf", memmap=True)


def test_saveload__nonequidistant(test_da_nonequidistant):
    idf.save("nonequidistant.idf", test_da_nonequidistant)
    assert Path("nonequidistant.idf").exists()
    da = idf.load("nonequidistant.idf")
    assert isinstance(da, xr.DataArray)
    assert np.array_equal(da, test_da_nonequidistant)
    # since the coordinates are created in float64 and stored in float32,
    # we lose some precision, which we have to allow for here
    xr.testing.assert_allclose(da, test_da_nonequidistant)


def test_lazy():
    """
    Reading should be lazily executed. That means it has to be part of the dask
    graph. Specifcally, the delayed function is imod.idf._read.

    This does the job of testing whether that function is part the graph.
    """
    a, _ = idf.dask("test.idf")
    assert "_read" in str(next(a.dask.items())[1])


def test_save_topbot__single_layer(test_da):
    da = test_da
    da.attrs["top"] = 1.0
    da.attrs["bot"] = 0.0
    idf.save("test", da)
    _, attrs = idf.read("test.idf")
    assert attrs["top"] == 1.0
    assert attrs["bot"] == 0.0


def test_save_topbot__layers(test_layerda):
    da = test_layerda
    da.attrs["top"] = range(1, 6)
    da.attrs["bot"] = range(5)
    idf.save("layer", da)
    _, attrs = idf.read("layer_l1.idf")
    assert attrs["top"] == 1.0
    assert attrs["bot"] == 0.0
    _, attrs = idf.read("layer_l2.idf")
    assert attrs["top"] == 2.0
    assert attrs["bot"] == 1.0


def test_save_topbot__errors(test_layerda):
    da = test_layerda
    da.attrs["top"] = 1
    da.attrs["bot"] = 0
    with pytest.raises(AssertionError):
        idf.save("layer", da)
    da.attrs["top"] = [1, 2]
    da.attrs["bot"] = [0, 1]
    with pytest.raises(AssertionError):
        idf.save("layer", da)
    da.attrs["top"] = ["a", "b"]
    da.attrs["bot"] = ["c", "d"]
    with pytest.raises(ValueError):
        idf.save("layer", da)
    da.attrs["top"] = [1, 2]
    da.attrs["bot"] = [0, 1, 3]
    with pytest.raises(AssertionError):
        idf.save("layer", da)


def test_has_dim():
    t = cftime.DatetimeProlepticGregorian(2019, 2, 28)
    assert idf._has_dim([t, 2, 3])
    assert not idf._has_dim([None, None, None])
    with pytest.raises(ValueError):
        idf._has_dim([t, 2, None])
