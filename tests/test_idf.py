from glob import glob
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from imod import idf


@pytest.fixture(scope="module")
def test_da(request):
    arr = np.ones((3, 4), dtype=np.float32)
    cellwidth = 1.0
    cellheight = cellwidth
    xmin = 0.0
    ymax = 3.0
    attrs = OrderedDict()
    attrs["res"] = (cellwidth, cellheight)
    attrs["transform"] = (cellwidth, 0.0, xmin, 0.0, -cellheight, ymax)
    kwargs = {
        "name": "test",
        "dims": ("y", "x"),  # only two dimensions in a single IDF
        "attrs": attrs,
    }

    def remove():
        try:
            os.remove("test.idf")
        except FileNotFoundError:
            pass

    request.addfinalizer(remove)
    return xr.DataArray(arr, **kwargs)


@pytest.fixture(scope="module")
def test_layerda(request):
    nlay = 5
    nrow = 3
    ncol = 4
    data = np.ones((nlay, nrow, ncol))
    dims = ("layer", "y", "x")
    coords = {
        "layer": np.arange(nlay) + 1,
        "y": np.arange(nrow, 0.0, -1.0) - 0.5,
        "x": np.arange(ncol) + 0.5,
    }

    def remove():
        paths = glob("layer_l[0-9].idf")
        for p in paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    request.addfinalizer(remove)
    return xr.DataArray(data, coords, dims)


def test_saveload(test_da):
    idf.save("test.idf", test_da)
    assert Path("test.idf").exists()
    da = idf.load("test.idf")
    assert isinstance(da, xr.DataArray)
    assert (test_da == da).all()


def test_lazy():
    """
    Reading should be lazily executed. That means it has to be part of the dask
    graph. Specifcally, the delayed function is imod.idf._read.

    This does the job of testing whether that function is part the graph.
    """
    a, attrs = idf.dask("test.idf")
    assert "function _read" in str(next(a.dask.items())[1][0])


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
