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
