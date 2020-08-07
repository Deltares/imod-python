import filecmp
import glob
import logging
import os
import pathlib
import textwrap
import time

import joblib
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

    river.to_netcdf(river_ncpath)
    river.to_zarr(river_zarrpath)
    # river.to_zarr(zarr.ZipStore(river_zarrzippath, mode="w"))

    # Test kwargs also
    chunks = {"time": 1, "layer": 1, "y": 3, "x": 4}
    imod.wq.River.from_file(river_ncpath, chunks=chunks)
    imod.wq.River.from_file(river_zarrpath, chunks=chunks)
    # imod.wq.River.from_file(river_zarrzippath, chunks=chunks)


def test_cached_river__max_n(test_timelayerda, tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(tmp_path / "caching-max_n.log", mode="w")
    logger.addHandler(fh)

    da = test_timelayerda
    river = imod.wq.River(stage=da, conductance=da, bottom_elevation=da, density=da)
    riverpath = tmp_path / "river.nc"
    river.to_netcdf(riverpath)
    nlayer, nrow, ncol = test_timelayerda.isel(time=0).shape
    expected = river._max_active_n("conductance", nlayer, nrow, ncol)

    my_cache = tmp_path / "my-cache"
    cached_river = imod.wq.River.from_file(riverpath, my_cache, 2)
    cached_river._filehashes["riv"] = cached_river._filehashself

    cache_path = tmp_path / "my-cache/imod/wq/caching/_max_n"
    # First round, cache is still empty.
    assert cache_path.exists()
    # First time, runs code
    actual1 = cached_river._max_active_n("conductance", nlayer, nrow, ncol)
    actual2 = cached_river._max_active_n("conductance", nlayer, nrow, ncol)

    # Recreate object
    cached_river = imod.wq.River.from_file(riverpath, my_cache, 2)
    cached_river._filehashes["riv"] = cached_river._filehashself
    actual3 = cached_river._max_active_n("conductance", nlayer, nrow, ncol)

    # release netcdf
    # Change river
    time.sleep(2.0)  # sleep two seconds so the modification time is different
    cached_river._dataset.close()
    river.to_netcdf(riverpath)
    cached_river = imod.wq.River.from_file(riverpath, my_cache, 2)
    cached_river._filehashes["riv"] = cached_river._filehashself
    actual4 = cached_river._max_active_n("conductance", nlayer, nrow, ncol)
    actual4 = cached_river._max_active_n("conductance", nlayer, nrow, ncol)

    assert actual1 == actual2 == actual3 == actual4 == expected

    fh.close()
    with open(tmp_path / "caching-max_n.log") as f:
        actual_log = f.read()

    expected_log = textwrap.dedent(
        """\
    MAX_N: Input is new. Counting anew.
    MAX_N: Input recognized. Skipping.
    MAX_N: Input recognized. Skipping.
    MAX_N: Input is new. Counting anew.
    MAX_N: Input recognized. Skipping.
    """
    )
    assert actual_log == expected_log


def test_cached_river__check(test_timelayerda, tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(tmp_path / "caching-check.log", mode="w")
    logger.addHandler(fh)

    da = test_timelayerda
    river = imod.wq.River(stage=da, conductance=da, bottom_elevation=da, density=da)
    riverpath = tmp_path / "river.nc"
    river.to_netcdf(riverpath)
    river._pkgcheck()

    # Only tests whether it runs without erroring
    my_cache = tmp_path / "my-cache"
    cached_river = imod.wq.River.from_file(riverpath, my_cache, 2)
    cached_river._filehashes["riv"] = cached_river._filehashself
    # this should result in only a single call.
    cached_river._pkgcheck()
    cached_river._pkgcheck()

    time.sleep(2.0)  # sleep two seconds so the modification time is different
    cached_river._dataset.close()
    river.to_netcdf(riverpath)
    cached_river = imod.wq.River.from_file(riverpath, my_cache, 2)
    cached_river._pkgcheck()
    cached_river._pkgcheck()

    fh.close()
    with open(tmp_path / "caching-check.log") as f:
        actual_log = f.read()

    expected_log = textwrap.dedent(
        """\
    CHECK: Input is new. Checking anew.
    CHECK: Input recognized. Skipping.
    CHECK: Input is new. Checking anew.
    CHECK: Input recognized. Skipping.
    """
    )
    assert actual_log == expected_log


def test_cached_river__save(test_timelayerda, tmp_path):
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(tmp_path / "caching-save.log", mode="w")
    logger.addHandler(fh)

    da = test_timelayerda
    riverpath = tmp_path / "river.nc"
    river = imod.wq.River(stage=da, conductance=da, bottom_elevation=da, density=da)
    # Default save for checking
    river.to_netcdf(riverpath)

    my_cache = tmp_path / "my-cache"
    cached_river = imod.wq.River.from_file(riverpath, my_cache, 2)
    cached_river._filehashes["riv"] = cached_river._filehashself
    cached_river._reldir = pathlib.Path(".")

    river.save(tmp_path / "basic-riv")
    ref_path = str(tmp_path / "basic-riv/**/*.idf")

    # SAVING: Input is new. Saving anew.
    # Call render to generate the list of _outputfiles
    cached_river._render(
        directory=tmp_path / "cached-riv",
        globaltimes=cached_river["time"].values,
        system_index=1,
        nlayer=5,
    )
    cached_river.save(tmp_path / "cached-riv")
    output_path = str(tmp_path / "cached-riv/**/*.idf")

    basic_files = [pathlib.Path(p) for p in glob.glob(ref_path, recursive=True)]
    caching_files = [pathlib.Path(p) for p in glob.glob(output_path, recursive=True)]
    assert set(p.name for p in basic_files) == set(p.name for p in caching_files)

    # SAVING: Input recognized.
    # SAVING: Output recognized. Skipping.
    cached_river.save(tmp_path / "cached-riv")

    # Now remove a single file, this should trigger a recompute.
    # SAVING: Input recognized.
    # SAVING: Output has been changed. Saving anew.
    os.remove(caching_files[0])
    cached_river.save(tmp_path / "cached-riv")
    caching_files = [pathlib.Path(p) for p in glob.glob(output_path, recursive=True)]
    assert set(p.name for p in basic_files) == set(p.name for p in caching_files)

    # SAVING: Input is new. Saving anew.
    time.sleep(2.0)  # sleep two seconds so the modification time is different
    cached_river._dataset.close()
    river.to_netcdf(riverpath)
    cached_river = imod.wq.River.from_file(riverpath, my_cache, 2)
    cached_river._render(
        directory=tmp_path / "cached-riv",
        globaltimes=cached_river["time"].values,
        system_index=1,
        nlayer=5,
    )
    cached_river._filehashes["riv"] = cached_river._filehashself
    cached_river._reldir = pathlib.Path(".")
    cached_river.save(tmp_path / "cached-riv")
    assert set(p.name for p in basic_files) == set(p.name for p in caching_files)

    fh.close()
    with open(tmp_path / "caching-save.log") as f:
        actual_log = f.read()

    expected_log = textwrap.dedent(
        """\
    SAVING: Input is new. Saving anew.
    SAVING: Input recognized.
    SAVING: Output recognized. Skipping.
    SAVING: Input recognized.
    SAVING: Output has been changed. Saving anew.
    SAVING: Input is new. Saving anew.
    """
    )
    assert actual_log == expected_log


# keep track of performance using pytest-benchmark?
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
    m1.write(tmp_path / "henry-basic")


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
    my_cache = tmp_path / "my-cache"
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


def henry_write_cache_zarr(tmp_path):
    bas, lpf, btn, adv, dsp, vdf, wel, pcg, gcg, oc = henry_input()
    # Store all data
    bas.to_netcdf(tmp_path / "bas.nc")
    lpf.to_netcdf(tmp_path / "lpf.nc")
    btn.to_netcdf(tmp_path / "btn.nc")
    adv.to_netcdf(tmp_path / "adv.nc")
    dsp.to_netcdf(tmp_path / "dsp.nc")
    wel.to_netcdf(tmp_path / "wel.nc")
    # Now load again
    my_cache = tmp_path / "my-cache"
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


def test_cached_model_write(tmp_path):
    # It seems like pytest-benchmark cannot be used here succesfully
    # the issue is the combination with tmp_path, and the existence
    # of the joblib.my_cache and netCDF files.
    henry_write(tmp_path)
    henry_write_cache(tmp_path)
    # Test whether written files are the same.
    assert filecmp.dircmp(tmp_path / "henry-basic", tmp_path / "henry-caching")
    # Speedup should be atleast a factor two in this case
    # it gets bigger with more data
    # This needs a benchmarking solution
