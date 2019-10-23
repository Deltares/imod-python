import numpy as np
import pytest
import rasterio
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
    da = xr.DataArray(data, **kwargs)
    da.attrs["crs"] = rasterio.crs.CRS(init="EPSG:28992")
    return da


def test_saveopen(test_da, tmp_path):
    da = test_da
    imod.rasterio.save(tmp_path / "withcrs.tif", da)
    daback = imod.rasterio.open(tmp_path / "withcrs.tif").load()
    daback.name = "test"
    assert daback.identical(test_da)

    da.attrs["crs"] = None
    imod.rasterio.save(tmp_path / "no_crs.asc", da)
    daback = imod.rasterio.open(tmp_path / "no_crs.asc").load()
    daback.name = "test"
    assert daback.identical(da)


def test_saveopen__nodata_dtype(test_da, tmp_path):
    da = test_da.copy()
    da[...] = np.nan
    imod.rasterio.save(tmp_path / "onlynan_f32", da, dtype=np.float32)
    daback = imod.rasterio.open(tmp_path / "onlynan_f32.tif").load()
    assert daback.dtype == np.float32
    assert (daback.isnull()).all()

    imod.rasterio.save(tmp_path / "onlynone_f32", da, nodata=None, dtype=np.float32)
    daback = imod.rasterio.open(tmp_path / "onlynone_f32.tif").load()
    assert daback.dtype == np.float32
    assert (daback.isnull()).all()

    imod.rasterio.save(tmp_path / "onlyminus1_f32", da, nodata=-1, dtype=np.float32)
    daback = imod.rasterio.open(tmp_path / "onlyminus1_f32.tif").load()
    # -1 is set as nodata, and converted into nan when reading into DataArrays
    # for floating point datatypes
    assert daback.dtype == np.float32
    assert (daback.isnull()).all()

    imod.rasterio.save(tmp_path / "onlyminus1_f64", da, nodata=-1, dtype=np.float64)
    daback = imod.rasterio.open(tmp_path / "onlyminus1_f64.tif").load()
    # -1 is set as nodata, and converted into nan when reading into DataArrays
    # for floating point datatypes
    assert daback.dtype == np.float64
    assert (daback.isnull()).all()

    imod.rasterio.save(tmp_path / "onlyminus1_i32", da, nodata=-1, dtype=np.int32)
    daback = imod.rasterio.open(tmp_path / "onlyminus1_i32.tif").load()
    assert daback.dtype == np.int32
    assert (daback == -1).all()
