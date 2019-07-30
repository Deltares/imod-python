import contextlib
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


def globremove(globpath):
    paths = glob.glob(globpath)
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


@pytest.fixture(scope="module")
def test_da(request):
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)

    def remove():
        globremove("test*.idf")

    request.addfinalizer(remove)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_da__nodxdy(request):
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = {"y": np.arange(ymax, ymin, dy), "x": np.arange(xmin, xmax, dx)}
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)

    def remove():
        globremove("testnodxdy.idf")

    request.addfinalizer(remove)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_nptimeda(request):
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["time"] = pd.date_range("2000-01-01", "2000-01-10", freq="D").values
    ntime = len(coords["time"])
    kwargs = {"name": "testnptime", "coords": coords, "dims": ("time", "y", "x")}
    data = np.ones((ntime, nrow, ncol), dtype=np.float32)

    def remove():
        globremove("testnptime*.idf")

    request.addfinalizer(remove)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_cftimeda(request):
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

    def remove():
        globremove("testcftime*.idf")

    request.addfinalizer(remove)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_layerda(request):
    nlay, nrow, ncol = 5, 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["layer"] = np.arange(nlay) + 1
    kwargs = {"name": "layer", "coords": coords, "dims": ("layer", "y", "x")}
    data = np.ones((nlay, nrow, ncol), dtype=np.float32)

    def remove():
        paths = glob.glob("layer_l[0-9].idf")
        for p in paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    request.addfinalizer(remove)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_timelayerda(request):
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

    def remove():
        paths = glob.glob("timelayer*.idf")
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
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "nonequidistant", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)

    def remove():
        try:
            os.remove("nonequidistant.idf")
        except FileNotFoundError:
            pass

    request.addfinalizer(remove)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_da_subdomains(request):
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

    def remove():
        globremove("subdomains_*.idf")

    request.addfinalizer(remove)
    return das


def test_open_subdomains(test_da_subdomains):
    subdomains = test_da_subdomains

    for i, subdomain in enumerate(subdomains):
        for layer, da in subdomain.groupby("layer"):
            idf.write(f"subdomains_l{layer}_p00{i}.idf", da)

    da = idf.open_subdomains("subdomains_*.idf", pattern=r"{name}_l{layer}_p\d+")

    assert np.all(da == 1.0)
    assert len(da.x) == 8
    assert len(da.y) == 6

    coords = util._xycoords((0.0, 8.0, 0.0, 6.0), (1.0, -1.0))
    assert np.all(da["y"].values == coords["y"])
    assert np.all(da["x"].values == coords["x"])

    assert isinstance(da, xr.DataArray)


def test_open_subdomains_error(test_da_subdomains):
    subdomains = test_da_subdomains

    for i, subdomain in enumerate(subdomains):
        for layer, da in subdomain.groupby("layer"):
            idf.write(f"subdomains_l{layer}_p00{i}.idf", da)

    # Add an additional subdomain with only one layer
    idf.write("subdomains_l1_p010.idf", subdomain.sel(layer=1))

    with pytest.raises(ValueError):
        da = idf.open_subdomains("subdomains_*.idf", pattern=r"{name}_l{layer}_p\d+")


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


def test_save__error(test_da):
    with pytest.raises(ValueError):
        idf.save("test.idf", test_da)


def test_saveopen__steady(test_da):
    first = test_da.copy().assign_coords(layer=1)
    second = test_da.copy().assign_coords(layer=2)
    steady_layers = xr.concat([first, second], dim="layer")
    steady_layers = steady_layers.assign_coords(time="steady-state")
    steady_layers = steady_layers.expand_dims("time")
    idf.save("test", steady_layers)
    da = idf.open("test_steady-state_l*.idf")
    assert da.identical(steady_layers)


def test_to_nan():
    a = np.array([1.0, 2.000001, np.nan, 4.0])
    c = idf._to_nan(a, np.nan)
    assert np.allclose(c, a, equal_nan=True)
    c = idf._to_nan(a, 2.0)
    b = np.array([1.0, np.nan, np.nan, 4.0])
    assert np.allclose(c, b, equal_nan=True)


def test_saveopen(test_da):
    idf.save("test", test_da)
    assert pathlib.Path("test.idf").exists()
    da = idf.open("test.idf")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_da)
    da = idf.open(pathlib.Path("test.idf"))
    assert da.identical(test_da)


def test_saveopen__paths(test_da, tmp_path):
    print("tmp_path:", tmp_path)
    idf.save("test", test_da)
    # open paths as a list of str
    da = idf.open(["test.idf"])
    assert da.identical(test_da)
    # open paths as a list of pathlib.Path
    da = idf.open([pathlib.Path("test.idf")])
    assert da.identical(test_da)
    # open nonexistent path
    with pytest.raises(FileNotFoundError):
        idf.open("nonexistent.idf")
    # open a file that is not an idf
    with open(tmp_path / "no.idf", "w") as f:
        f.write("not the IDF header you expect")
    with pytest.raises(ValueError):
        idf.open(tmp_path / "no.idf")


def test_save__int32coords(test_da__nodxdy):
    test_da = test_da__nodxdy
    test_da.x.values = test_da.x.values.astype(np.int32)
    test_da.y.values = test_da.y.values.astype(np.int32)
    idf.save("testnodxdy", test_da)
    assert pathlib.Path("testnodxdy.idf").exists()


def test_saveopen__nptime(test_nptimeda):
    idf.save("testnptime", test_nptimeda)
    da = idf.open("testnptime*.idf")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_nptimeda)


def test_saveopen__cftime_withinbounds(test_nptimeda):
    cftimes = []
    for time in test_nptimeda.time.values:
        dt = pd.Timestamp(time).to_pydatetime()
        cftimes.append(cftime.DatetimeProlepticGregorian(*dt.timetuple()[:6]))
    da = idf.open("testnptime*.idf", use_cftime=True)
    assert isinstance(da, xr.DataArray)
    assert all(np.array(cftimes) == da.time.values)


def test_saveopen__cftime_outofbounds(test_cftimeda):
    idf.save("testcftime", test_cftimeda)
    with pytest.warns(UserWarning):
        da = idf.open("testcftime*.idf")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_cftimeda)


def test_saveopen__cftime_nodim(test_cftimeda):
    da = test_cftimeda.copy().isel(time=0)
    da.name = "testcftime-nodim"
    idf.save("testcftime-nodim", da)
    loaded = idf.open("testcftime-nodim*.idf", use_cftime=True).squeeze("time").load()
    assert da.identical(loaded)


def test_saveopen_sorting_headers_paths(test_timelayerda):
    idf.save("timelayer", test_timelayerda)
    loaded = idf.open("timelayer_*.idf").isel(x=0, y=0).values.ravel()
    assert np.allclose(np.sort(loaded), loaded)


def test_saveopen_timelayer(test_timelayerda):
    idf.save("timelayer", test_timelayerda)
    da = idf.open("timelayer_*.idf")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_timelayerda)


def test_saveopen__nonequidistant(test_da_nonequidistant):
    idf.save("nonequidistant", test_da_nonequidistant)
    assert pathlib.Path("nonequidistant.idf").exists()
    da = idf.open("nonequidistant.idf")
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
    a, _ = idf._dask("test.idf")
    try:  # dask 2.0
        assert "_read" in str(a.dask.items()[0][1])
    # TODO: Remove when dask 2.0 is commonly installed
    except TypeError:  # dask < 2.0
        assert "_read" in str(next(a.dask.items())[1])

    # Test whether a change directory doesn't mess up reading a file
    @contextlib.contextmanager
    def remember_cwd():
        """
        from:
        https://stackoverflow.com/questions/169070/how-do-i-write-a-decorator-that-restores-the-cwd
        """
        curdir = os.getcwd()
        try:
            yield
        finally:
            os.chdir(curdir)

    with remember_cwd():
        os.chdir("..")
        a.compute()


def test_save_topbot__single_layer(test_da):
    da = test_da
    da = da.assign_coords(z=0.5)
    da = da.assign_coords(dz=1.0)
    idf.save("test", da)
    _, attrs = idf.read("test.idf")
    assert attrs["top"] == 1.0
    assert attrs["bot"] == 0.0


def test_save_topbot__layers(test_layerda):
    da = test_layerda
    da = da.assign_coords(z=("layer", np.arange(1.0, 6.0) - 0.5))
    idf.save("layer", da)
    _, attrs = idf.read("layer_l1.idf")
    assert attrs["top"] == 1.0
    assert attrs["bot"] == 0.0
    _, attrs = idf.read("layer_l2.idf")
    assert attrs["top"] == 2.0
    assert attrs["bot"] == 1.0
    # Read multiple idfs
    actual = idf.open("layer_l*.idf")
    assert np.allclose(actual["z"], da["z"])


def test_save_topbot__layers_nonequidistant(test_layerda):
    da = test_layerda
    dz = np.arange(-1.0, -6.0, -1.0)
    z = np.cumsum(dz) - 0.5 * dz
    da = da.assign_coords(z=("layer", z))
    da = da.assign_coords(dz=("layer", dz))
    idf.save("layer", da)
    # Read multiple idfs
    actual = idf.open("layer_l*.idf")
    assert np.allclose(actual["z"], da["z"])
    assert np.allclose(actual["dz"], da["dz"])


def test_save_topbot__only_z(test_layerda):
    da = test_layerda
    da = da.assign_coords(z=("layer", np.arange(1.0, 6.0) - 0.5))
    da = da.swap_dims({"layer": "z"})
    da = da.drop("layer")
    idf.save("layer", da)
    _, attrs = idf.read("layer_l1.idf")
    assert attrs["top"] == 1.0
    assert attrs["bot"] == 0.0
    _, attrs = idf.read("layer_l2.idf")
    assert attrs["top"] == 2.0
    assert attrs["bot"] == 1.0

    actual = idf.open("layer_l1.idf")
    assert float(actual["z"]) == 0.5


def test_save_topbot__errors(test_layerda):
    da = test_layerda
    # non-equidistant, cannot infer dz
    z = np.array([0.0, -1.0, -3.0, -4.5, -5.0])
    da = da.assign_coords(z=("layer", z))
    with pytest.raises(ValueError):
        idf.save("layer", da)


def test_has_dim():
    t = cftime.DatetimeProlepticGregorian(2019, 2, 28)
    assert idf._has_dim([t, 2, 3])
    assert not idf._has_dim([None, None, None])
    with pytest.raises(ValueError):
        idf._has_dim([t, 2, None])


def test_check_cellsizes():
    # (h["dx"], h["dy"])
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.000001, 3.0])
    c = np.array([1.0, 2.1, 3.0])
    d = np.array([2.0, 2.000001, 2.0])
    e = np.array([4.0, 5.0, 6.0])
    f = np.array([4.0, 5.0, 6.0, 7.0])
    # length one always checks out
    idf._check_cellsizes([(2.0, 3.0)])
    # floats only
    idf._check_cellsizes([(2.0, 3.0), (2.0, 3.0)])
    idf._check_cellsizes([(2.0, 3.0), (2.000001, 3.0)])
    # ndarrays only
    idf._check_cellsizes([(a, e), (a, e)])
    # different length a and f
    idf._check_cellsizes([(a, f), (a, f)])
    idf._check_cellsizes([(a, e), (b, e)])
    # mix of floats and ndarrays
    idf._check_cellsizes([(2.0, d)])
    with pytest.raises(ValueError, match="Cellsizes of IDFs do not match"):
        # floats only
        idf._check_cellsizes([(2.0, 3.0), (2.1, 3.0)])
        # ndarrays only
        idf._check_cellsizes([(a, e), (c, e)])
        # mix of floats and ndarrays
        idf._check_cellsizes([(2.1, d)])
        # Unequal lengths
        idf._check_cellsizes([(a, e), (f, e)])
