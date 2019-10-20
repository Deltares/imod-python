import glob
import os
import pathlib

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="module")
def test_da():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_nptimeda():
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["time"] = pd.date_range("2000-01-01", "2000-01-10", freq="D").values
    ntime = len(coords["time"])
    kwargs = {"name": "testnptime", "coords": coords, "dims": ("time", "y", "x")}
    data = np.ones((ntime, nrow, ncol), dtype=np.float32)
    return xr.DataArray(data, **kwargs)


# TODO: decide on functionality, reimplement test
# def test_rasterio_write_read(test_da, tmp_path):
#    imod.rasterio.write(tmp_path / "raster.asc", test_da)
#    da_back = imod.rasterio.open(tmp_path / "raster.asc")
#    assert da_back.dims == ("y", "x")
#    assert len(da_back.nodatavals) == 1
#    assert np.isnan(da_back.nodatavals[0])
#    assert (da_back == 1).all().item()
#    assert test_da.dtype == np.float32
#    assert not test_da.identical(da_back)
#    assert not test_da.equals(da_back)
#
#    print(test_da)
#    print(da_back)
#
#    assert not test_da.broadcast_equals(da_back)
#    imod.rasterio.write(
#        tmp_path / "raster_9999.asc", test_da, driver="AAIGrid", nodata=-9999
#    )
#    da_back_9999 = imod.rasterio.open(tmp_path / "raster_9999.asc")
#    assert len(da_back_9999.nodatavals) == 1
#    assert da_back_9999.nodatavals[0] == -9999.0
#    assert not da_back_9999.identical(da_back)
#    assert da_back_9999.equals(da_back)
#    assert da_back_9999.broadcast_equals(da_back)
#    assert da_back_9999.dtype == "float32"
#    # getting a heap corruption fatal error on this... nevermind pcraster
#    # imod.rasterio.write(
#    #     tmp_path / "raster.map", test_da.astype(np.bool), driver="PCRaster", nodata=0
#    # )
#    # da_back_map = xr.open_rasterio(tmp_path / "raster.map")
#    # assert da_back_map.dtype == "uint8"
#    # assert len(da_back_map.nodatavals) == 1
#    # assert da_back_map.nodatavals[0] == 255


def test_rasterio_read(test_nptimeda, tmp_path):
    # create a list of tifs with timestamps by converting idf to tif
    # needed since we don't yet support reading and combining at once
    # like idf.read
    imod.idf.save(tmp_path / "rastertime", test_nptimeda)
    idf_paths = list(tmp_path.glob("rastertime_*.idf"))
    for idf_path in idf_paths:
        da = imod.idf.open(idf_path)
        imod.rasterio.write(tmp_path / (idf_path.stem + ".tif"), da.squeeze("time"))

    # now we read all timestamps at once
    da_back = imod.rasterio.open(tmp_path / "rastertime_*.tif")
    # some extra attributes set by xarray.open_rasterio remain
    assert not da_back.identical(test_nptimeda)
    assert da_back.equals(test_nptimeda)
