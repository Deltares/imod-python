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
    assert actual.identical(expected)


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
    assert actual.identical(expected)


def test_interpolate_2d():
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
    assert actual.identical(expected)


def test_interpolate_2d__reversed_y():
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
    assert actual.identical(expected)
