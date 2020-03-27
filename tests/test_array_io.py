import glob
import os
import pathlib

import cftime
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr

import imod
from imod import util
from imod import array_io
from imod.array_io import reading


@pytest.fixture(scope="module", params=[np.float32, np.float64])
def test_da(request):
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=request.param)
    da = xr.DataArray(data, **kwargs)
    return da


@pytest.fixture(scope="module", params=[np.float32, np.float64])
def test_da__nodxdy(request):
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = {"y": np.arange(ymax, ymin, dy), "x": np.arange(xmin, xmax, dx)}
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=request.param)
    da = xr.DataArray(data, **kwargs)
    return da


@pytest.fixture(scope="module", params=[np.float32, np.float64])
def test_nptimeda(request):
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["time"] = pd.date_range("2000-01-01", "2000-01-10", freq="D").values
    ntime = len(coords["time"])
    kwargs = {"name": "testnptime", "coords": coords, "dims": ("time", "y", "x")}
    data = np.ones((ntime, nrow, ncol), dtype=request.param)
    da = xr.DataArray(data, **kwargs)
    return da


@pytest.fixture(scope="module", params=[np.float32, np.float64])
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
    data = np.ones((ntime, nrow, ncol), dtype=request.param)
    da = xr.DataArray(data, **kwargs)
    return da


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
    da = xr.DataArray(data, **kwargs)
    return da


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

    da = xr.DataArray(data, **kwargs)
    return da


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

    da = xr.DataArray(data, **kwargs)
    return da


def test_to_nan():
    a = np.array([1.0, 2.000001, np.nan, 4.0])
    c = reading._to_nan(a, np.nan)
    assert np.allclose(c, a, equal_nan=True)
    c = reading._to_nan(a, 2.0)
    b = np.array([1.0, np.nan, np.nan, 4.0])
    assert np.allclose(c, b, equal_nan=True)


def test_check_cellsizes():
    # (h["dx"], h["dy"])
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.000001, 3.0])
    c = np.array([1.0, 2.1, 3.0])
    d = np.array([2.0, 2.000001, 2.0])
    e = np.array([4.0, 5.0, 6.0])
    f = np.array([4.0, 5.0, 6.0, 7.0])
    # length one always checks out
    reading._check_cellsizes([(2.0, 3.0)])
    # floats only
    reading._check_cellsizes([(2.0, 3.0), (2.0, 3.0)])
    reading._check_cellsizes([(2.0, 3.0), (2.000001, 3.0)])
    # ndarrays only
    reading._check_cellsizes([(a, e), (a, e)])
    # different length a and f
    reading._check_cellsizes([(a, f), (a, f)])
    reading._check_cellsizes([(a, e), (b, e)])
    # mix of floats and ndarrays
    reading._check_cellsizes([(2.0, d)])
    with pytest.raises(ValueError, match="Cellsizes of IDFs do not match"):
        # floats only
        reading._check_cellsizes([(2.0, 3.0), (2.1, 3.0)])
        # ndarrays only
        reading._check_cellsizes([(a, e), (c, e)])
        # mix of floats and ndarrays
        reading._check_cellsizes([(2.1, d)])
        # Unequal lengths
        reading._check_cellsizes([(a, e), (f, e)])


def test_has_dim():
    t = cftime.DatetimeProlepticGregorian(2019, 2, 28)
    assert reading._has_dim([t, 2, 3])
    assert not reading._has_dim([None, None, None])
    with pytest.raises(ValueError):
        reading._has_dim([t, 2, None])


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen__steady(test_da, tmp_path, kind):
    module, ext = kind
    first = test_da.copy().assign_coords(layer=1)
    second = test_da.copy().assign_coords(layer=2)
    steady_layers = xr.concat([first, second], dim="layer")
    steady_layers = steady_layers.assign_coords(time="steady-state")
    steady_layers = steady_layers.expand_dims("time")
    module.save(tmp_path / "test", steady_layers)
    da = module.open(tmp_path / f"test_steady-state_l*.{ext}")
    assert da.identical(steady_layers)


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen(test_da, kind, tmp_path):
    module, ext = kind
    module.save(tmp_path / f"test", test_da)
    assert (tmp_path / f"test.{ext}").exists()
    da = module.open(tmp_path / f"test.{ext}")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_da)
    da = module.open(tmp_path / f"test.{ext}")
    assert da.identical(test_da)


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen__descending_layer(test_layerda, kind, tmp_path):
    module, ext = kind
    # Flip around test_layerda
    flip = slice(None, None, -1)
    test_layerda = test_layerda.copy().isel(layer=flip)
    module.save(tmp_path / f"layer", test_layerda)
    assert (tmp_path / f"layer_l1.{ext}").exists()
    da = module.open(tmp_path / f"layer_l*.{ext}")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_layerda.isel(layer=flip))


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen__paths(test_da, kind, tmp_path):
    module, ext = kind
    module.save(tmp_path / "test", test_da)
    # open paths as a list of str
    da = module.open([str(tmp_path / f"test.{ext}")])
    assert da.identical(test_da)
    # open paths as a list of pathlib.Path
    da = module.open([tmp_path / f"test.{ext}"])
    assert da.identical(test_da)
    # open nonexistent path
    with pytest.raises(FileNotFoundError):
        module.open(tmp_path / f"nonexistent.{ext}")
    # open a file that is not an idf
    with open(tmp_path / f"no.{ext}", "w") as f:
        f.write("not the IDF header you expect")

    if ext == "idf":
        with pytest.raises(ValueError):
            module.open(tmp_path / f"no.{ext}")
    elif ext == "tif":
        with pytest.raises(rasterio.errors.RasterioIOError):
            module.open(tmp_path / f"no.{ext}")


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_save__int32coords(test_da__nodxdy, kind, tmp_path):
    module, ext = kind
    test_da = test_da__nodxdy
    test_da.coords["x"] = test_da.coords["x"].astype(np.int32)
    test_da.coords["y"] = test_da.coords["y"].astype(np.int32)
    module.save(tmp_path / "testnodxdy", test_da)
    assert (tmp_path / f"testnodxdy.{ext}").exists()


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen__nptime(test_nptimeda, kind, tmp_path):
    module, ext = kind
    module.save(tmp_path / "testnptime", test_nptimeda)
    da = module.open(tmp_path / f"testnptime*.{ext}")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_nptimeda)


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen__cftime_withinbounds(test_nptimeda, kind, tmp_path):
    module, ext = kind
    cftimes = []
    for time in test_nptimeda.time.values:
        dt = pd.Timestamp(time).to_pydatetime()
        cftimes.append(cftime.DatetimeProlepticGregorian(*dt.timetuple()[:6]))
    module.save(tmp_path / "testnptime", test_nptimeda)
    da = module.open(tmp_path / f"testnptime*.{ext}", use_cftime=True)
    assert isinstance(da, xr.DataArray)
    assert all(np.array(cftimes) == da.time.values)


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen__cftime_outofbounds(test_cftimeda, kind, tmp_path):
    module, ext = kind
    module.save(tmp_path / "testcftime", test_cftimeda)
    with pytest.warns(UserWarning):
        da = module.open(tmp_path / f"testcftime*.{ext}")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_cftimeda)


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen__cftime_nodim(test_cftimeda, kind, tmp_path):
    module, ext = kind
    da = test_cftimeda.copy().isel(time=0)
    da.name = "testcftime-nodim"
    module.save(tmp_path / "testcftime-nodim", da)
    loaded = (
        module.open(tmp_path / f"testcftime-nodim*.{ext}", use_cftime=True)
        .squeeze("time")
        .load()
    )
    assert da.identical(loaded)


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen_sorting_headers_paths(test_timelayerda, kind, tmp_path):
    module, ext = kind
    module.save(tmp_path / "timelayer", test_timelayerda)
    loaded = module.open(tmp_path / f"timelayer_*.{ext}").isel(x=0, y=0).values.ravel()
    assert np.allclose(np.sort(loaded), loaded)


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen_timelayer(test_timelayerda, kind, tmp_path):
    module, ext = kind
    module.save(tmp_path / "timelayer", test_timelayerda)
    da = module.open(tmp_path / f"timelayer_*.{ext}")
    assert isinstance(da, xr.DataArray)
    assert da.identical(test_timelayerda)


@pytest.mark.parametrize("kind", [(imod.rasterio, "tif"), (imod.idf, "idf")])
def test_saveopen_timelayer_chunks(test_timelayerda, kind, tmp_path):
    chunkda = test_timelayerda.chunk({"layer": 1, "time": 1})
    module, ext = kind
    module.save(tmp_path / "chunked", chunkda)
    da = module.open(tmp_path / f"chunked_*.{ext}")
    # .identical() fails in this case ...
    assert isinstance(da, xr.DataArray)
    assert da.equals(chunkda)


def test_lazy(test_da, tmp_path):
    """
    Reading should be lazily executed. That means it has to be part of the dask
    graph. Specifcally, the delayed function is imod.module._read.

    This does the job of testing whether that function is part the graph.
    """
    imod.idf.save(tmp_path / "test", test_da)
    a, _ = array_io.reading._dask(
        tmp_path / "test.idf", _read=imod.idf._read, header=imod.idf.header
    )
    try:  # dask 2.0
        assert "_read" in str(a.dask.items()[0][1])
    # TODO: Remove when dask 2.0 is commonly installed
    except TypeError:  # dask < 2.0
        assert "_read" in str(next(a.dask.items())[1])

    with util.cd(".."):
        a.compute()
