import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import imod
from imod import util


@pytest.fixture(scope="module")
def test_timelayerda():
    ntime, nlay, nrow, ncol = 3, 5, 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["layer"] = np.arange(nlay) + 8
    coords["time"] = pd.date_range("2000-01-01", "2002-01-01", freq="YS").values

    kwargs = {
        "name": "timelayer",
        "coords": coords,
        "dims": ("time", "layer", "y", "x"),
    }
    data = np.ones((ntime, nlay, nrow, ncol), dtype=np.float32)
    da = xr.DataArray(data, **kwargs)
    return da


def test_from_file(test_timelayerda, tmp_path):
    da = test_timelayerda
    river = imod.wq.River(stage=da, conductance=da, bottom_elevation=da, density=da)
    river_ncpath = tmp_path / "river.nc"
    river_zarrpath = tmp_path / "river.zarr"
    # TODO: zip fails on CI for some reason?
    # river_zarrzippath = tmp_path / "river.zip"

    river.dataset.to_netcdf(river_ncpath)
    river.dataset.to_zarr(river_zarrpath)
    # river.to_zarr(zarr.ZipStore(river_zarrzippath, mode="w"))

    # Test kwargs also
    chunks = {"time": 1, "layer": 1, "y": 3, "x": 4}
    imod.wq.River.from_file(river_ncpath, chunks=chunks)
    imod.wq.River.from_file(river_zarrpath, chunks=chunks)
    # imod.wq.River.from_file(river_zarrzippath, chunks=chunks)
