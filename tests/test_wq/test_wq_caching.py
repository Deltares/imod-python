import glob
import pathlib
import os
import time

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
    riverpath = tmp_path / "river.nc"
    river.to_netcdf(riverpath)
    expected = river._max_active_n("conductance", 2)

    memory = joblib.Memory(tmp_path / "my-cache")
    cached_river = imod.wq.River.from_file(riverpath, memory)
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
    cached_river = imod.wq.River.from_file(riverpath, memory)
    cached_river._filehashes["riv"] = cached_river._filehashself
    actual3 = cached_river._max_active_n("conductance", 2)
    assert len(os.listdir(cache_path)) == 2

    # Delete cached_river to release netcdf
    del cached_river
    river.to_netcdf(riverpath)
    cached_river = imod.wq.River.from_file(riverpath, memory)
    cached_river._filehashes["riv"] = cached_river._filehashself
    actual4 = cached_river._max_active_n("conductance", 2)
    # A new hash should've been created since the file has been modified.
    assert len(os.listdir(cache_path)) == 3

    assert actual1 == actual2 == actual3 == actual4 == expected


def test_cached_river__check(test_timelayerda, tmp_path):
    da = test_timelayerda
    river = imod.wq.River(stage=da, conductance=da, bottom_elevation=da, density=da,)
    riverpath = tmp_path / "river.nc"
    river.to_netcdf(riverpath)
    river._pkgcheck()

    # Only tests whether it runs without erroring
    memory = joblib.Memory(tmp_path / "my-cache")
    cached_river = imod.wq.River.from_file(riverpath, memory)
    cached_river._filehashes["riv"] = cached_river._filehashself
    cached_river._pkgcheck()


def test_cached_river__save(test_timelayerda, tmp_path):
    da = test_timelayerda
    riverpath = tmp_path / "river.nc"
    river = imod.wq.River(stage=da, conductance=da, bottom_elevation=da, density=da,)
    # Default save for checking
    river.to_netcdf(riverpath)

    memory = joblib.Memory(tmp_path / "my-cache")
    cached_river = imod.wq.River.from_file(riverpath, memory)
    cached_river._filehashes["riv"] = cached_river._filehashself
    cached_river._reldir = pathlib.Path(".")

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
    river.to_netcdf(riverpath)
    cached_river = imod.wq.River.from_file(riverpath, memory)
    cached_river._render(
        directory=tmp_path / "cached-riv",
        globaltimes=cached_river["time"].values,
        system_index=1,
    )
    cached_river._filehashes["riv"] = cached_river._filehashself
    cached_river._reldir = pathlib.Path(".")
    cached_river.save(tmp_path / "cached-riv")
    assert len(os.listdir(cache_path)) == 3
    assert set(p.name for p in basic_files) == set(p.name for p in caching_files)


# keep track of performance using pytest-benchmark
# https://pytest-benchmark.readthedocs.io/en/stable/
def henry_input():
    # Discretization1
    nrow = 200
    ncol = 20  # made larger for noticeable
    nlay = 50

    dz = 1.0
    dx = 1.0
    dy = -dx

    # scale parameters with discretization
    qscaled = 0.03 * (dz * abs(dy))

    # Setup ibound
    bnd = xr.DataArray(
        data=np.full((nlay, nrow, ncol), 1.0),
        coords={
            "y": np.arange(0.5 * dy, dy * nrow, dy),
            "x": np.arange(0.5 * dx, dx * ncol, dx),
            "layer": np.arange(1, 1 + nlay),
            "dx": dx,
            "dy": dy,
        },
        dims=("layer", "y", "x"),
    )

    top1D = xr.DataArray(
        np.arange(nlay * dz, 0.0, -dz), {"layer": np.arange(1, 1 + nlay)}, ("layer")
    )
    bot = top1D - 1.0

    bas = imod.wq.BasicFlow(ibound=bnd, top=50.0, bottom=bot, starting_head=1.0)
    lpf = imod.wq.LayerPropertyFlow(
        k_horizontal=10.0, k_vertical=10.0, specific_storage=0.0
    )
    btn = imod.wq.BasicTransport(icbund=bnd, starting_concentration=35.0, porosity=0.35)
    adv = imod.wq.AdvectionTVD(courant=1.0)
    dsp = imod.wq.Dispersion(longitudinal=0.1, diffusion_coefficient=1.0e-9)
    vdf = imod.wq.VariableDensityFlow(density_concentration_slope=0.71)
    wel = imod.wq.Well(id_name="well", x=0.5, y=0.5, rate=qscaled)
    pcg = imod.wq.PreconditionedConjugateGradientSolver(
        max_iter=150, inner_iter=30, hclose=0.0001, rclose=1.0, relax=0.98, damp=1.0
    )
    gcg = imod.wq.GeneralizedConjugateGradientSolver(
        max_iter=150,
        inner_iter=30,
        cclose=1.0e-6,
        preconditioner="mic",
        lump_dispersion=True,
    )

    oc = imod.wq.OutputControl(save_head_idf=True, save_concentration_idf=True)

    return bas, lpf, btn, adv, dsp, vdf, wel, pcg, gcg, oc


def henry_write(tmp_path):
    bas, lpf, btn, adv, dsp, vdf, wel, pcg, gcg, oc = henry_input()
    m1 = imod.wq.SeawatModel("HenryCase")
    m1["bas"] = bas
    m1["lpf"] = lpf
    m1["btn"] = btn
    m1["wel"] = wel
    m1["adv"] = adv
    m1["dsp"] = dsp
    m1["vdf"] = vdf
    m1["oc"] = oc
    m1["pcg"] = pcg
    m1["gcg"] = gcg
    m1.time_discretization(times=["2000-01-01", "2001-01-01"])

    # If this seems crude, it is.
    # However, pytest-benchmark is very difficult to get to work
    starttime = time.time()
    for i in range(10):
        m1.write(tmp_path / "henry-caching")
    endtime = time.time()
    return endtime - starttime


def henry_write_cache(tmp_path):
    bas, lpf, btn, adv, dsp, vdf, wel, pcg, gcg, oc = henry_input()
    # Store all data
    bas.to_netcdf(tmp_path / "bas.nc")
    lpf.to_netcdf(tmp_path / "lpf.nc")
    btn.to_netcdf(tmp_path / "btn.nc")
    adv.to_netcdf(tmp_path / "adv.nc")
    dsp.to_netcdf(tmp_path / "dsp.nc")
    wel.to_netcdf(tmp_path / "wel.nc")
    # Now load again
    my_cache = joblib.Memory(tmp_path / "my-cache")
    bas2 = imod.wq.BasicFlow.from_file(tmp_path / "bas.nc", my_cache)
    lpf2 = imod.wq.LayerPropertyFlow.from_file(tmp_path / "lpf.nc", my_cache)
    btn2 = imod.wq.BasicTransport.from_file(tmp_path / "btn.nc", my_cache)
    dsp2 = imod.wq.Dispersion.from_file(tmp_path / "dsp.nc", my_cache)
    wel2 = imod.wq.Well.from_file(tmp_path / "wel.nc", my_cache)
    # Caching case
    m2 = imod.wq.SeawatModel("HenryCase")
    m2["bas"] = bas2
    m2["lpf"] = lpf2
    m2["btn"] = btn2
    m2["wel"] = wel2
    m2["adv"] = adv
    m2["dsp"] = dsp2
    m2["vdf"] = vdf
    m2["oc"] = oc
    m2["pcg"] = pcg
    m2["gcg"] = gcg
    m2.time_discretization(times=["2000-01-01", "2001-01-01"])

    m2.write(tmp_path / "henry-caching")

    starttime = time.time()
    for i in range(10):
        m2.write("henry-caching")
    endtime = time.time()
    return endtime - starttime


def test_cached_model_write(tmp_path):
    # It seems like pytest-benchmark cannot be used here succesfully
    # the issue is the combination with tmp_path, and the existence
    # of the joblib.Memory and netCDF files.
    t1 = henry_write(tmp_path)
    t2 = henry_write_cache(tmp_path)
    # Speedup should be atleast a factor two in this case
    # it gets bigger with more data
    assert (t1 / t2) > 2
