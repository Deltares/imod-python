import glob
import os
import pathlib

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod import idf
from imod import util
from imod import array_io


@pytest.fixture(scope="module")
def test_da():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_da__nodxdy():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = {"y": np.arange(ymax, ymin, dy), "x": np.arange(xmin, xmax, dx)}
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_nptimeda():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["time"] = pd.date_range("2000-01-01", "2000-01-10", freq="D").values
    ntime = len(coords["time"])
    kwargs = {"name": "testnptime", "coords": coords, "dims": ("time", "y", "x")}
    data = np.ones((ntime, nrow, ncol), dtype=np.float32)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_cftimeda():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["time"] = [
        cftime.DatetimeProlepticGregorian(y, 1, 1) for y in range(1000, 10_000, 1000)
    ]
    ntime = len(coords["time"])
    kwargs = {"name": "testcftime", "coords": coords, "dims": ("time", "y", "x")}
    data = np.ones((ntime, nrow, ncol), dtype=np.float32)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_layerda():
    nlay, nrow, ncol = 5, 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["layer"] = np.arange(nlay) + 1
    kwargs = {"name": "layer", "coords": coords, "dims": ("layer", "y", "x")}
    data = np.ones((nlay, nrow, ncol), dtype=np.float32)
    return xr.DataArray(data, **kwargs)


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
    for i in range(ntime):
        for j, layer in enumerate(range(8, 8 + nlay)):
            data[i, j, ...] = layer * (i + 1)

    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_speciestimelayerda():
    nspecies, ntime, nlay, nrow, ncol = 2, 3, 2, 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["layer"] = np.arange(nlay) + 8
    coords["time"] = pd.date_range("2000-01-01", "2002-01-01", freq="YS").values
    coords["species"] = np.arange(nspecies) + 1

    kwargs = {
        "name": "conc",
        "coords": coords,
        "dims": ("species", "time", "layer", "y", "x"),
    }
    data = np.ones((nspecies, ntime, nlay, nrow, ncol), dtype=np.float32)
    for s in range(nspecies):
        for i in range(ntime):
            for j, layer in enumerate(range(8, 8 + nlay)):
                data[s, i, j, ...] = layer * (i + 1)

    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_da_nonequidistant():
    nrow, ncol = 3, 4
    dx = np.array([0.9, 1.1, 0.8, 1.2])
    dy = np.array([-1.3, -0.7, -1.0])
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "nonequidistant", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)

    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_da_subdomains():
    nlayer, nrow, ncol = (3, 4, 5)
    dx, dy = (1.0, -1.0)
    layer = [1, 2, 3]
    xmin = (0.0, 3.0, 3.0, 0.0)
    xmax = (5.0, 8.0, 8.0, 5.0)
    ymin = (0.0, 2.0, 0.0, 2.0)
    ymax = (4.0, 6.0, 4.0, 6.0)
    data = np.ones((nlayer, nrow, ncol), dtype=np.float32)

    kwargs = {"name": "subdomains", "dims": ("layer", "y", "x")}

    das = []
    for subd_extent in zip(xmin, xmax, ymin, ymax):
        kwargs["coords"] = util._xycoords(subd_extent, (dx, dy))
        kwargs["coords"]["layer"] = layer
        das.append(xr.DataArray(data, **kwargs))

    return das


def test_open_subdomains(test_da_subdomains, tmp_path):
    subdomains = test_da_subdomains

    for i, subdomain in enumerate(subdomains):
        for layer, da in subdomain.groupby("layer"):
            idf.write(tmp_path / f"subdomains_20000101_l{layer}_p00{i}.idf", da)

    da = idf.open_subdomains(tmp_path / "subdomains_*.idf")

    assert np.all(da == 1.0)
    assert len(da.x) == 8
    assert len(da.y) == 6

    coords = util._xycoords((0.0, 8.0, 0.0, 6.0), (1.0, -1.0))
    assert np.all(da["y"].values == coords["y"])
    assert np.all(da["x"].values == coords["x"])

    assert isinstance(da, xr.DataArray)


def test_open_subdomains_error(test_da_subdomains, tmp_path):
    subdomains = test_da_subdomains

    for i, subdomain in enumerate(subdomains):
        for layer, da in subdomain.groupby("layer"):
            idf.write(tmp_path / f"subdomains_20000101_l{layer}_p00{i}.idf", da)

    # Add an additional subdomain with only one layer
    idf.write(tmp_path / "subdomains_20000101_l1_p010.idf", subdomain.sel(layer=1))

    with pytest.raises(ValueError):
        da = idf.open_subdomains(tmp_path / "subdomains_*.idf")


def test_open_speciestimelayer(test_speciestimelayerda, tmp_path):
    idf.save(
        tmp_path / "conc",
        test_speciestimelayerda,
        pattern=r"{name}_{time:%Y%m%d%H%M%S}_c{species}_l{layer}{extension}",
    )  # Right now save does not have a default way of saving species,
    # since the imodwq default is quite bad (requires time)
    da = idf.open(tmp_path / "conc*.idf")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_speciestimelayerda)


def test_xycoords_equidistant():
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    assert np.allclose(coords["x"], np.arange(xmin + dx / 2.0, xmax, dx))
    assert np.allclose(coords["y"], np.arange(ymax + dy / 2.0, ymin, dy))
    assert coords["dx"] == dx
    assert coords["dy"] == dy


def test_xycoords_nonequidistant():
    dx = np.array([0.9, 1.1, 0.8, 1.2])
    dy = np.array([-1.3, -0.7, -1.0])
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    assert np.allclose(coords["x"], np.array([0.45, 1.45, 2.4, 3.4]))
    assert np.allclose(coords["y"], np.array([2.35, 1.35, 0.5]))
    assert coords["dx"][0] == "x"
    assert np.allclose(coords["dx"][1], dx)
    assert coords["dy"][0] == "y"
    assert np.allclose(coords["dy"][1], dy)


def test_xycoords_equidistant_array():
    dx = np.array([2.0, 2.0, 2.0, 2.0])
    dy = np.array([-0.5, -0.500001, -0.5])
    xmin, xmax = 0.0, 8.0
    ymin, ymax = 0.0, 1.5
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    assert np.allclose(coords["x"], np.arange(xmin + 1.0, xmax, 2.0))
    assert np.allclose(coords["y"], np.arange(ymax - 0.25, ymin, -0.5))
    assert coords["dx"] == 2.0
    assert coords["dy"] == -0.5


def test_saveopen__steady(test_da, tmp_path):
    first = test_da.copy().assign_coords(layer=1)
    second = test_da.copy().assign_coords(layer=2)
    steady_layers = xr.concat([first, second], dim="layer")
    steady_layers = steady_layers.assign_coords(time="steady-state")
    steady_layers = steady_layers.expand_dims("time")
    idf.save(tmp_path / "test", steady_layers)
    da = idf.open(tmp_path / "test_steady-state_l*.idf")
    assert da.identical(steady_layers)


def test_saveopen(test_da, tmp_path):
    idf.save(tmp_path / "test", test_da)
    assert (tmp_path / "test.idf").exists()
    da = idf.open(tmp_path / "test.idf")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_da)
    da = idf.open(tmp_path / "test.idf")
    assert da.identical(test_da)


def test_saveopen__paths(test_da, tmp_path):
    idf.save(tmp_path / "test", test_da)
    # open paths as a list of str
    da = idf.open([str(tmp_path / "test.idf")])
    assert da.identical(test_da)
    # open paths as a list of pathlib.Path
    da = idf.open([tmp_path / "test.idf"])
    assert da.identical(test_da)
    # open nonexistent path
    with pytest.raises(FileNotFoundError):
        idf.open(tmp_path / "nonexistent.idf")
    # open a file that is not an idf
    with open(tmp_path / "no.idf", "w") as f:
        f.write("not the IDF header you expect")
    with pytest.raises(ValueError):
        idf.open(tmp_path / "no.idf")


def test_save__int32coords(test_da__nodxdy, tmp_path):
    test_da = test_da__nodxdy
    test_da.x.values = test_da.x.values.astype(np.int32)
    test_da.y.values = test_da.y.values.astype(np.int32)
    idf.save(tmp_path / "testnodxdy", test_da)
    assert (tmp_path / "testnodxdy.idf").exists()


def test_saveopen__nptime(test_nptimeda, tmp_path):
    idf.save(tmp_path / "testnptime", test_nptimeda)
    da = idf.open(tmp_path / "testnptime*.idf")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_nptimeda)


def test_saveopen__cftime_withinbounds(test_nptimeda, tmp_path):
    cftimes = []
    for time in test_nptimeda.time.values:
        dt = pd.Timestamp(time).to_pydatetime()
        cftimes.append(cftime.DatetimeProlepticGregorian(*dt.timetuple()[:6]))
    idf.save(tmp_path / "testnptime", test_nptimeda)
    da = idf.open(tmp_path / "testnptime*.idf", use_cftime=True)
    assert isinstance(da, xr.DataArray)
    assert all(np.array(cftimes) == da.time.values)


def test_saveopen__cftime_outofbounds(test_cftimeda, tmp_path):
    idf.save(tmp_path / "testcftime", test_cftimeda)
    with pytest.warns(UserWarning):
        da = idf.open(tmp_path / "testcftime*.idf")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_cftimeda)


def test_saveopen__cftime_nodim(test_cftimeda, tmp_path):
    da = test_cftimeda.copy().isel(time=0)
    da.name = "testcftime-nodim"
    idf.save(tmp_path / "testcftime-nodim", da)
    loaded = (
        idf.open(tmp_path / "testcftime-nodim*.idf", use_cftime=True)
        .squeeze("time")
        .load()
    )
    assert da.identical(loaded)


def test_saveopen_sorting_headers_paths(test_timelayerda, tmp_path):
    idf.save(tmp_path / "timelayer", test_timelayerda)
    loaded = idf.open(tmp_path / "timelayer_*.idf").isel(x=0, y=0).values.ravel()
    assert np.allclose(np.sort(loaded), loaded)


def test_saveopen_timelayer(test_timelayerda, tmp_path):
    idf.save(tmp_path / "timelayer", test_timelayerda)
    da = idf.open(tmp_path / "timelayer_*.idf")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_timelayerda)


def test_saveopen__nonequidistant(test_da_nonequidistant, tmp_path):
    idf.save(tmp_path / "nonequidistant", test_da_nonequidistant)
    assert (tmp_path / "nonequidistant.idf").exists()
    da = idf.open(tmp_path / "nonequidistant.idf")
    assert isinstance(da, xr.DataArray)
    assert np.array_equal(da, test_da_nonequidistant)
    # since the coordinates are created in float64 and stored in float32,
    # we lose some precision, which we have to allow for here
    xr.testing.assert_allclose(da, test_da_nonequidistant)


def test_lazy(test_da, tmp_path):
    """
    Reading should be lazily executed. That means it has to be part of the dask
    graph. Specifcally, the delayed function is imod.idf._read.

    This does the job of testing whether that function is part the graph.
    """
    idf.save(tmp_path / "test", test_da)
    a, _ = array_io.reading._dask(
        tmp_path / "test.idf", _read=idf._read, header=idf.header
    )
    try:  # dask 2.0
        assert "_read" in str(a.dask.items()[0][1])
    # TODO: Remove when dask 2.0 is commonly installed
    except TypeError:  # dask < 2.0
        assert "_read" in str(next(a.dask.items())[1])

    with util.cd(".."):
        a.compute()


def test_save_topbot__single_layer(test_da, tmp_path):
    da = test_da
    da = da.assign_coords(z=0.5)
    da = da.assign_coords(dz=1.0)
    idf.save(tmp_path / "test", da)
    _, attrs = idf.read(tmp_path / "test.idf")
    assert attrs["top"] == 1.0
    assert attrs["bot"] == 0.0


def test_save_topbot__layers(test_layerda, tmp_path):
    da = test_layerda
    da = da.assign_coords(z=("layer", np.arange(1.0, 6.0) - 0.5))
    idf.save(tmp_path / "layer", da)
    _, attrs = idf.read(tmp_path / "layer_l1.idf")
    assert attrs["top"] == 1.0
    assert attrs["bot"] == 0.0
    _, attrs = idf.read(tmp_path / "layer_l2.idf")
    assert attrs["top"] == 2.0
    assert attrs["bot"] == 1.0
    # Read multiple idfs
    actual = idf.open(tmp_path / "layer_l*.idf")
    assert np.allclose(actual["z"], da["z"])


def test_save_topbot__layers_nonequidistant(test_layerda, tmp_path):
    da = test_layerda
    dz = np.arange(-1.0, -6.0, -1.0)
    z = np.cumsum(dz) - 0.5 * dz
    da = da.assign_coords(z=("layer", z))
    da = da.assign_coords(dz=("layer", dz))
    idf.save(tmp_path / "layer", da)
    # Read multiple idfs
    actual = idf.open(tmp_path / "layer_l*.idf")
    assert np.allclose(actual["z"], da["z"])
    assert np.allclose(actual["dz"], da["dz"])


def test_save_topbot__only_z(test_layerda, tmp_path):
    da = test_layerda
    da = da.assign_coords(z=("layer", np.arange(1.0, 6.0) - 0.5))
    da = da.swap_dims({"layer": "z"})
    da = da.drop("layer")
    idf.save(tmp_path / "layer", da)
    _, attrs = idf.read(tmp_path / "layer_l1.idf")
    assert attrs["top"] == 1.0
    assert attrs["bot"] == 0.0
    _, attrs = idf.read(tmp_path / "layer_l2.idf")
    assert attrs["top"] == 2.0
    assert attrs["bot"] == 1.0

    actual = idf.open(tmp_path / "layer_l1.idf")
    assert float(actual["z"]) == 0.5


def test_save_topbot__errors(test_layerda, tmp_path):
    da = test_layerda
    # non-equidistant, cannot infer dz
    z = np.array([0.0, -1.0, -3.0, -4.5, -5.0])
    da = da.assign_coords(z=("layer", z))
    with pytest.raises(ValueError):
        idf.save(tmp_path / "layer", da)
