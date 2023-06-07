from pathlib import Path

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
    da.attrs.pop("crs")
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


# Note: GDAL is especially idiosyncratic when it comes to floats without a
# decimal. It will write the first value as a float (e.g. 10.0), but all
# subsequent values are written as integer, presumably to save space. Numpy
# savetxt obviously does not replicate this behavior. Concretely, I've omitted
# floats without a decimal in these test cases.
@pytest.mark.parametrize(
    "value,nodata,dtype,precision,digits",
    [
        (1, -999, np.int32, None, None),
        (1.1, -999.0, np.float32, None, None),
        (1.1, np.nan, np.float32, None, None),
        (1.1, -999.9, np.float32, None, None),
        (1.1, -999.99, np.float32, None, None),
        (1.11, -999.99, np.float32, None, None),
        # Decimal precision
        (1, -999, np.int32, 3, None),
        (1.1, -999.99, np.float32, 2, None),
        (1.1, np.nan, np.float32, 2, None),
        # Significant digits
        (1, -999, np.int32, None, 3),
        (1.1, -999.99, np.float32, None, 4),
        (1.1, np.nan, np.float32, None, 5),
    ],
)
def test_rasterio_ascii(test_da, tmp_path, value, nodata, dtype, precision, digits):
    def write_rasterio(path, a, profile):
        with rasterio.Env():
            with rasterio.open(path, "w", **profile) as ds:
                ds.write(a, 1)
        return

    def assert_equal_content(path_a, path_b):
        assert Path(path_a).exists()
        assert Path(path_b).exists()
        with open(path_a, "r") as f:
            content_a = f.read()
        with open(path_b, "r") as f:
            content_b = f.read()
        assert content_a == content_b

    da = xr.full_like(test_da, value)
    profile = {
        "transform": imod.util.transform(da),
        "driver": "AAIGrid",
        "height": da.y.size,
        "width": da.x.size,
        "count": 1,
        "dtype": dtype,
        "nodata": nodata,
    }
    if precision is not None:
        profile["decimal_precision"] = precision
    if digits is not None:
        profile["significant_digits"] = digits

    path_a = tmp_path / "expected.asc"
    path_b = tmp_path / "actual.asc"

    write_rasterio(path_a, da.values, profile)
    imod.rasterio.write_aaigrid(path_b, da.values, profile)

    assert_equal_content(path_a, path_b)
