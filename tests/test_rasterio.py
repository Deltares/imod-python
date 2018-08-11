import pytest
import os
import imod
import rasterio
import numpy as np
import xarray as xr
from affine import Affine
from glob import glob


def remove(globpath):
    for p in glob(globpath):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


@pytest.fixture(scope="module")
def write_tif(request):
    def _write_tif(path, epsg, dtype=np.float64, rotation_angle=None):
        nrow = 5
        ncol = 8
        values = (10 * np.random.rand(nrow, ncol)).astype(dtype)

        profile = dict()
        profile["crs"] = rasterio.crs.CRS.from_epsg(epsg)
        profile["transform"] = Affine.translation(20.0, 30.0)
        if rotation_angle:
            profile["transform"] = Affine.rotation(rotation_angle)
        profile["driver"] = "GTiff"
        profile["height"] = nrow
        profile["width"] = ncol
        profile["count"] = 1
        profile["dtype"] = dtype

        with rasterio.Env():
            with rasterio.open(path, "w", **profile) as ds:
                ds.write(values, 1)

    def teardown():
        remove("*.tif")

    request.addfinalizer(teardown)
    return _write_tif


def test_basic_resample__nearest(write_tif):
    """Test nearest neighbour upsampling to halved cellsize"""
    write_tif("basic.tif", epsg=28992)

    da = xr.open_rasterio("basic.tif").squeeze("band")
    dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(da)
    data = np.empty((10, 16))
    coords = {
        "y": np.linspace(ymin + 0.25 * dy, ymax - 0.25 * dy, 10),
        "x": np.linspace(xmin + 0.25 * dx, xmax - 0.25 * dx, 16),
    }
    dims = ("y", "x")
    like = xr.DataArray(data, coords, dims)
    newda = imod.rasterio.resample(da, like, method="nearest")

    newarr = np.empty((10, 16))
    with rasterio.open("basic.tif") as src:
        arr = src.read()
        aff = src.transform
        crs = src.crs
        newaff = Affine(aff.a / 2.0, aff.b, aff.c, aff.d, aff.e / 2.0, aff.f)

    rasterio.warp.reproject(
        arr,
        newarr,
        src_transform=aff,
        dst_transform=newaff,
        src_crs=crs,
        dst_crs=crs,
        resampling=rasterio.warp.Resampling.nearest,
    )

    np.allclose(newda.values, newarr)


def test_basic_resample__bilinear(write_tif):
    """Test bilinear upsampling to halved cellsize"""
    write_tif("basic.tif", epsg=28992)

    da = xr.open_rasterio("basic.tif").squeeze("band")
    dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(da)
    data = np.empty((10, 16))
    coords = {
        "y": np.linspace(ymin + 0.25 * dy, ymax - 0.25 * dy, 10),
        "x": np.linspace(xmin + 0.25 * dx, xmax - 0.25 * dx, 16),
    }
    dims = ("y", "x")
    like = xr.DataArray(data, coords, dims)
    newda = imod.rasterio.resample(da, like, method="bilinear")

    newarr = np.empty((10, 16))
    with rasterio.open("basic.tif") as src:
        arr = src.read()
        aff = src.transform
        crs = src.crs
        newaff = Affine(aff.a / 2.0, aff.b, aff.c, aff.d, aff.e / 2.0, aff.f)

    rasterio.warp.reproject(
        arr,
        newarr,
        src_transform=aff,
        dst_transform=newaff,
        src_crs=crs,
        dst_crs=crs,
        resampling=rasterio.warp.Resampling.bilinear,
    )

    np.allclose(newda.values, newarr)


def test_basic_reproject(write_tif):
    """Basic reprojection from EPSG:28992 to EPSG:32631"""
    write_tif("basic.tif", epsg=28992)
    dst_crs = "+init=EPSG:32631"
    da = xr.open_rasterio("basic.tif").squeeze("band")
    newda = imod.rasterio.resample(da, src_crs="+init=EPSG:28992", dst_crs=dst_crs)

    with rasterio.open("basic.tif") as src:
        arr = src.read()
        src_transform = src.transform
        src_crs = src.crs
        src_height = src.height
        src_width = src.width

    src_crs = rasterio.crs.CRS(src_crs)
    bounds = rasterio.transform.array_bounds(src_height, src_width, src_transform)
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *bounds
    )

    newarr = np.empty((dst_height, dst_width))

    rasterio.warp.reproject(
        arr,
        newarr,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.nearest,
    )

    np.allclose(newda.values, newarr)


def test_reproject__use_src_attrs(write_tif):
    """Reprojection from EPSG:28992 to EPSG:32631, using on attrs generated by xarray."""
    write_tif("basic.tif", epsg=28992)
    dst_crs = "+init=EPSG:32631"
    da = xr.open_rasterio("basic.tif").squeeze("band")
    newda = imod.rasterio.resample(da, dst_crs=dst_crs, use_src_attrs=True)

    with rasterio.open("basic.tif") as src:
        arr = src.read()
        src_transform = src.transform
        src_crs = src.crs
        src_height = src.height
        src_width = src.width

    src_crs = rasterio.crs.CRS(src_crs)
    bounds = rasterio.transform.array_bounds(src_height, src_width, src_transform)
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *bounds
    )

    newarr = np.empty((dst_height, dst_width))

    rasterio.warp.reproject(
        arr,
        newarr,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.nearest,
    )

    np.allclose(newda.values, newarr)


def test_reproject_resample(write_tif):
    """
    Reprojection from EPSG:28992 to EPSG:32631, using on attrs generated by xarray,
    then resample to a like DataArray.
    """
    write_tif("basic.tif", epsg=28992)
    da = xr.open_rasterio("basic.tif").squeeze("band")

    dst_crs = {"init":"EPSG:32631"}
    data = np.empty((10, 16))
    coords = {
        "y": np.linspace(5313578.75, 5313574.25, 10),
        "x": np.linspace(523420.25, 523427.75, 16),
    }
    dims = ("y", "x")
    like = xr.DataArray(data, coords, dims)
    newda = imod.rasterio.resample(da, like, dst_crs=dst_crs, use_src_attrs=True)

    with rasterio.open("basic.tif") as src:
        arr = src.read()
        src_transform = src.transform
        src_crs = src.crs

    dst_transform = Affine(0.50, 0.0, 523420.25, 0.0, -0.50, 5313578.75)
    newarr = np.empty((10, 16))
    rasterio.warp.reproject(
        arr,
        newarr,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.nearest,
    )

    assert ~np.isnan(newda.values).all()
    np.allclose(newda.values, newarr)


def test_reproject_rotation__via_kwargs(write_tif):
    """Reprojection from EPSG:28992 to EPSG:32631, by specifying kwarg"""
    write_tif("rotated.tif", epsg=28992, rotation_angle=45.0)
    dst_crs = {"init":"EPSG:32631"}
    da = xr.open_rasterio("rotated.tif").squeeze("band")
    src_transform = Affine.rotation(45.0)
    newda = imod.rasterio.resample(
        da,
        src_crs="+init=EPSG:28992",
        dst_crs=dst_crs,
        reproject_kwargs={"src_transform": src_transform},
    )

    with rasterio.open("rotated.tif") as src:
        arr = src.read()
        src_transform = src.transform
        src_crs = src.crs
        src_height = src.height
        src_width = src.width

    src_crs = rasterio.crs.CRS(src_crs)
    bounds = rasterio.transform.array_bounds(src_height, src_width, src_transform)
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *bounds
    )

    newarr = np.empty((dst_height, dst_width))

    rasterio.warp.reproject(
        arr,
        newarr,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.nearest,
    )

    np.allclose(newda.values, newarr)


def test_reproject_rotation__use_src_attrs(write_tif):
    """Reprojection from EPSG:28992 to EPSG:32631, using attrs generated by xarray."""
    write_tif("rotated.tif", epsg=28992, rotation_angle=45.0)
    dst_crs = "+init=EPSG:32631"
    da = xr.open_rasterio("rotated.tif").squeeze("band")
    newda = imod.rasterio.resample(da, dst_crs=dst_crs, use_src_attrs=True)

    with rasterio.open("rotated.tif") as src:
        arr = src.read()
        src_transform = src.transform
        src_crs = src.crs
        src_height = src.height
        src_width = src.width

    src_crs = rasterio.crs.CRS(src_crs)
    bounds = rasterio.transform.array_bounds(src_height, src_width, src_transform)
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *bounds
    )

    newarr = np.empty((dst_height, dst_width))

    rasterio.warp.reproject(
        arr,
        newarr,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.nearest,
    )

    np.allclose(newda.values, newarr)
