import glob
import pathlib
import os

import joblib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

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


def test_cached_river__max_n(test_timelayerda, tmp_path):
    da = test_timelayerda
    river = imod.wq.River(stage=da, conductance=da, bottom_elevation=da, density=da,)
    river.to_netcdf(tmp_path / "river.nc")
    expected = river._max_active_n("conductance", 2)

    memory = joblib.Memory(tmp_path / "my-cache")
    cached_river = imod.wq.caching(imod.wq.River, tmp_path / "river.nc", memory)
    cached_river._filehashes["riv"] = cached_river._filehashself

    cache_path = tmp_path / "my-cache/imod/wq/caching/_max_n"
    # First round, cache is still empty.
    assert cache_path.exists()
    assert len(os.listdir(cache_path)) == 0
    # First time, runs code
    actual1 = cached_river._max_active_n("conductance", 2)
    assert cache_path.exists()
    # a dir with a hash is created, and the function have been stored: a dir and a file.
    assert len(os.listdir(cache_path)) == 2
    actual2 = cached_river._max_active_n("conductance", 2)
    assert len(os.listdir(cache_path)) == 2

    # Recreate object
    cached_river = imod.wq.caching(imod.wq.River, tmp_path / "river.nc", memory)
    cached_river._filehashes["riv"] = cached_river._filehashself
    actual3 = cached_river._max_active_n("conductance", 2)
    assert len(os.listdir(cache_path)) == 2

    # Delete cached_river to release netcdf
    del cached_river
    river.to_netcdf(tmp_path / "river.nc")
    cached_river = imod.wq.caching(imod.wq.River, tmp_path / "river.nc", memory)
    cached_river._filehashes["riv"] = cached_river._filehashself
    actual4 = cached_river._max_active_n("conductance", 2)
    # A new hash should've been created since the file has been modified.
    assert len(os.listdir(cache_path)) == 3

    assert actual1 == actual2 == actual3 == actual4 == expected


def test_cached_river__save(test_timelayerda, tmp_path):
    da = test_timelayerda
    river = imod.wq.River(stage=da, conductance=da, bottom_elevation=da, density=da,)
    # Default save for checking
    river.to_netcdf(tmp_path / "river.nc")

    memory = joblib.Memory(tmp_path / "my-cache")
    cached_river = imod.wq.caching(imod.wq.River, tmp_path / "river.nc", memory)
    cached_river._filehashes["riv"] = cached_river._filehashself

    river.save(tmp_path / "basic-riv")
    cached_river.save(tmp_path / "cached-riv")
    # Call render to generate the list of _outputfiles
    cached_river._render(
        directory=tmp_path / "cached-riv",
        globaltimes=cached_river["time"].values,
        system_index=1,
    )
    cache_path = tmp_path / "my-cache/imod/wq/caching/_save"
    output_path = str(tmp_path / "cached-riv/**/*.idf")
    ref_path = str(tmp_path / "basic-riv/**/*.idf")
    assert len(os.listdir(cache_path)) == 2

    basic_files = [pathlib.Path(p) for p in glob.glob(ref_path, recursive=True)]
    caching_files = [pathlib.Path(p) for p in glob.glob(output_path, recursive=True)]
    assert set(p.name for p in basic_files) == set(p.name for p in caching_files)

    # Now remove a single file, this should trigger a recompute.
    os.remove(caching_files[0])
    cached_river.save(tmp_path / "cached-riv")
    caching_files = [pathlib.Path(p) for p in glob.glob(output_path, recursive=True)]
    assert set(p.name for p in basic_files) == set(p.name for p in caching_files)

    del cached_river
    river.to_netcdf(tmp_path / "river.nc")
    cached_river = imod.wq.caching(imod.wq.River, tmp_path / "river.nc", memory)
    cached_river._render(
        directory=tmp_path / "cached-riv",
        globaltimes=cached_river["time"].values,
        system_index=1,
    )
    cached_river._filehashes["riv"] = cached_river._filehashself
    cached_river.save(tmp_path / "cached-riv")
    assert len(os.listdir(cache_path)) == 3
    assert set(p.name for p in basic_files) == set(p.name for p in caching_files)
