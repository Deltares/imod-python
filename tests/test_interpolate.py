import os

import numba
import numpy as np
import pytest
import xarray as xr

import imod


def test_interpolate_1d():
    data = np.arange(5.0)
    x = np.arange(5.0)
    dst_x = np.array([2.5, 3.5])
    dims = ("x",)
    source = xr.DataArray(data, {"x": x}, dims)
    like = xr.DataArray([np.nan, np.nan], {"x": dst_x}, dims)
    interpolator_1d = imod.prepare.Regridder(method="multilinear")
    actual = interpolator_1d.regrid(source, like)
    expected = xr.full_like(like, [2.5, 3.5])
    # Note: xr.identical() fails, as the regridder as a dx coord for cellsizes.
    assert np.allclose(actual.values, expected.values, equal_nan=True)


def test_interpolate_1d__reversed():
    data = np.arange(5.0)[::-1]
    x = np.arange(5.0)[::-1]
    dst_x = np.array([2.5, 3.5])[::-1]
    dims = ("x",)
    source = xr.DataArray(data, {"x": x}, dims)
    like = xr.DataArray([np.nan, np.nan], {"x": dst_x}, dims)
    interpolator_1d = imod.prepare.Regridder(method="multilinear")
    actual = interpolator_1d.regrid(source, like)
    expected = xr.full_like(like, [3.5, 2.5])
    assert np.allclose(actual.values, expected.values, equal_nan=True)


def test_interpolate_1d__beyond_egdes():
    data = [0.0, 1.0]
    x = [0.5, 1.5]
    dst_x = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25]
    dims = ("x",)
    source = xr.DataArray(data, {"x": x}, dims)
    like_data = np.full(len(dst_x), np.nan)
    like = xr.DataArray(like_data, {"x": dst_x}, dims)
    interpolator_1d = imod.prepare.Regridder(method="multilinear")
    actual = interpolator_1d.regrid(source, like)
    expected = xr.full_like(like, [np.nan] * 3 + [0.0, 0.25, 0.75, 1.0, np.nan])
    assert np.allclose(actual.values, expected.values, equal_nan=True)


@pytest.mark.parametrize("chunksize", [1, 2])
def test_interpolate_2d(chunksize):
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    x = [0.5, 1.5]
    dst_x = [0.75, 1.25]
    y = [0.0, 1.0]
    dst_y = [0.25, 0.75]
    dims = ("y", "x")
    source = xr.DataArray(data, {"x": x, "y": y}, dims)
    dst_data = np.full_like(data, np.nan)
    like = xr.DataArray(dst_data, {"y": dst_y, "x": dst_x}, dims)
    interpolator_2d = imod.prepare.Regridder(method="multilinear")
    actual = interpolator_2d.regrid(source, like)
    expected = xr.full_like(like, [[0.25, 0.25], [0.75, 0.75]])
    assert np.allclose(actual.values, expected.values, equal_nan=True)

    # Now with chunks
    source = source.chunk({"x": chunksize, "y": chunksize})
    actual = interpolator_2d.regrid(source, like)
    assert np.allclose(actual.values, expected.values, equal_nan=True)


@pytest.mark.parametrize("chunksize", [1, 2])
def test_interpolate_2d__reversed_y(chunksize):
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    x = [0.5, 1.5]
    dst_x = [0.75, 1.25]
    y = list(reversed([0.0, 1.0]))
    dst_y = list(reversed([0.25, 0.75]))
    dims = ("y", "x")
    source = xr.DataArray(data, {"x": x, "y": y}, dims)
    dst_data = np.full_like(data, np.nan)
    like = xr.DataArray(dst_data, {"y": dst_y, "x": dst_x}, dims)
    interpolator_2d = imod.prepare.Regridder(method="multilinear")
    actual = interpolator_2d.regrid(source, like)
    expected = xr.full_like(like, [[0.25, 0.25], [0.75, 0.75]])
    assert np.allclose(actual.values, expected.values, equal_nan=True)

    # Now with chunks
    source = source.chunk({"x": chunksize})
    actual = interpolator_2d.regrid(source, like)
    assert np.allclose(actual.values, expected.values, equal_nan=True)


@pytest.mark.parametrize("chunksize", [1, 2])
def test_interpolate_1d__nan_withstartingedge(chunksize):
    data = [np.nan, 1.0, 1.0]
    x = [0.5, 1.5, 2.5]
    # 1.0 is on the starting edge, and should get a value
    dst_x = [0.25, 0.50, 0.75, 1.0]
    dims = ("x",)
    source = xr.DataArray(data, {"x": x}, dims)
    like_data = np.full(len(dst_x), np.nan)
    like = xr.DataArray(like_data, {"x": dst_x}, dims)
    interpolator_1d = imod.prepare.Regridder(method="multilinear")
    actual = interpolator_1d.regrid(source, like)
    expected = xr.full_like(like, [np.nan] * 3 + [1.0])
    assert np.allclose(actual.values, expected.values, equal_nan=True)

    # Now with chunks
    source = source.chunk({"x": chunksize})
    actual = interpolator_1d.regrid(source, like)
    assert np.allclose(actual.values, expected.values, equal_nan=True)


@pytest.mark.parametrize("chunksize", [1, 2])
def test_interpolate_1d__nan_withendingedge(chunksize):
    data = [1.0, 1.0, np.nan]
    x = [0.5, 1.5, 2.5]
    # 3.0 is on the starting edge, and should get a value
    dst_x = [1.0, 1.5, 2.0, 2.5]
    dims = ("x",)
    source = xr.DataArray(data, {"x": x}, dims)
    like_data = np.full(len(dst_x), np.nan)
    like = xr.DataArray(like_data, {"x": dst_x}, dims)
    interpolator_1d = imod.prepare.Regridder(method="multilinear")
    actual = interpolator_1d.regrid(source, like)
    expected = xr.full_like(like, [1.0] * 2 + [np.nan] * 2)
    assert np.allclose(actual.values, expected.values, equal_nan=True)

    # Now with chunks
    source = source.chunk({"x": chunksize})
    actual = interpolator_1d.regrid(source, like)
    assert np.allclose(actual.values, expected.values, equal_nan=True)


@pytest.mark.parametrize("chunksize", [1, 2])
def test_interpolate_2d__over_z(chunksize):
    data = np.array([[[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]]])
    x = [0.5, 1.5]
    dst_x = [0.75, 1.25]
    y = [0.0, 1.0]
    dst_y = [0.25, 0.75]
    z = [0.0, 1.0]
    dst_z = [0.0, 1.0]
    dims = ("z", "y", "x")
    source = xr.DataArray(data, {"x": x, "y": y, "z": z}, dims)
    dst_data = np.full_like(data, np.nan)
    like = xr.DataArray(dst_data, {"y": dst_y, "x": dst_x, "z": dst_z}, dims)
    interpolator_2d = imod.prepare.Regridder(method="multilinear")
    actual = interpolator_2d.regrid(source, like)
    expected = xr.full_like(
        like, [[[0.25, 0.25], [0.75, 0.75]], [[1.25, 1.25], [1.75, 1.75]]]
    )
    assert np.allclose(actual.values, expected.values, equal_nan=True)

    # Now with chunks
    source = source.chunk({"x": chunksize, "y": chunksize})
    actual = interpolator_2d.regrid(source, like)
    assert np.allclose(actual.values, expected.values, equal_nan=True)


@pytest.mark.parametrize("chunksize", [1, 2])
def test_interpolate_3d__over_xyz(chunksize):
    data = np.array([[[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]]])
    x = [0.5, 1.5]
    dst_x = [0.75, 1.25]
    y = [0.0, 1.0]
    dst_y = [0.25, 0.75]
    z = [0.0, 1.0]
    dst_z = [0.05, 0.95]
    dims = ("z", "y", "x")
    source = xr.DataArray(data, {"x": x, "y": y, "z": z}, dims)
    dst_data = np.full_like(data, np.nan)
    like = xr.DataArray(dst_data, {"y": dst_y, "x": dst_x, "z": dst_z}, dims)
    interpolator_3d = imod.prepare.Regridder(method="multilinear")
    actual = interpolator_3d.regrid(source, like)
    expected = xr.full_like(like, [[[0.3, 0.3], [0.8, 0.8]], [[1.2, 1.2], [1.7, 1.7]]])
    # This fails for some reason, different ordering of coords (not dims!):
    # assert actual.identical(expected)
    assert np.allclose(actual.values, expected.values, equal_nan=True)

    # Now with chunks
    source = source.chunk({"x": chunksize, "y": chunksize, "z": chunksize})
    actual = interpolator_3d.regrid(source, like)
    assert np.allclose(actual.values, expected.values, equal_nan=True)


# TODO: add tests for checking nodata behaviour in 2D and 3D
# TODO: maybe do parametrize stuff versus scipy linear interp?
