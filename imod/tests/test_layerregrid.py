import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="module")
def test_da():
    nlayer = 2
    nrow = 2
    ncol = 3
    shape = (nlayer, nrow, ncol)
    dims = ("layer", "y", "x")
    coords = {"layer": [1, 2], "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    data = np.arange(np.product(shape), dtype=np.float64).reshape(shape)
    source = xr.DataArray(data, coords, dims)
    return source


def test_layerregridder__mean_1(test_da):
    layerregridder = imod.prepare.LayerRegridder(method="mean")
    source = test_da
    top_src = xr.full_like(source, 0.0)
    top_src.data[1, :, :] = -2.0
    bottom_src = xr.full_like(source, -2.0)
    bottom_src.data[1, :, :] = -4.0

    top_dst = xr.full_like(source, 0.0)
    top_dst.data[1, :, :] = -3.0
    bottom_dst = xr.full_like(source, -3.0)
    bottom_dst.data[1, :, :] = -4.0

    actual = layerregridder.regrid(source, top_src, bottom_src, top_dst, bottom_dst)
    coords = {"layer": [1, 2], "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    dims = ("layer", "y", "x")
    expected = xr.DataArray(
        [[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]],
        coords,
        dims,
    )
    assert actual.identical(expected)


def test_layerregridder__mean_2(test_da):
    layerregridder = imod.prepare.LayerRegridder(method="mean")
    source = test_da
    top_src = xr.full_like(source, 0.0)
    top_src.data[1, :, :] = -2.0
    bottom_src = xr.full_like(source, -2.0)
    bottom_src.data[1, :, :] = -4.0

    coords = {"layer": [1, 2, 3], "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    dims = ("layer", "y", "x")
    top_dst = xr.DataArray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
            [[-3, -3, -3], [-3, -3, -3]],
        ],
        coords,
        dims,
    )
    bottom_dst = xr.full_like(top_dst, -1.0)
    bottom_dst.data[1, :, :] = -3.0
    bottom_dst.data[2, :, :] = -4.0

    actual = layerregridder.regrid(source, top_src, bottom_src, top_dst, bottom_dst)
    coords = {"layer": [1, 2, 3], "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    dims = ("layer", "y", "x")
    expected = xr.DataArray(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        coords,
        dims,
    )
    assert actual.identical(expected)


def test_layerregridder_dst_larger_src(test_da):
    layerregridder = imod.prepare.LayerRegridder(method="mean")
    source = test_da
    top_src = xr.full_like(source, 0.0)
    top_src.data[1, :, :] = -2.0
    bottom_src = xr.full_like(source, -2.0)
    bottom_src.data[1, :, :] = -4.0

    coords = {"layer": [1, 2, 3], "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    dims = ("layer", "y", "x")
    top_dst = xr.DataArray(
        [
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
            [[-3.0, -4.0, -3.0], [-3.0, -4.0, -3.0]],
        ],
        coords,
        dims,
    )
    bottom_dst = xr.DataArray(
        [
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
            [[-3.0, -4.0, -3.0], [-3.0, -4.0, -3.0]],
            [[-5.0, -5.0, -3], [-5.0, -5.0, -3]],
        ],
        coords,
        dims,
    )

    actual = layerregridder.regrid(source, top_src, bottom_src, top_dst, bottom_dst)
    coords = {"layer": [1, 2, 3], "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    dims = ("layer", "y", "x")
    expected = xr.DataArray(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[3.0, 5.0, 5.0], [6.0, 8.0, 8.0]],
            [[6.0, np.nan, np.nan], [9.0, np.nan, np.nan]],
        ],
        coords,
        dims,
    )
    assert actual.identical(expected)


def test_layerregridder__mode(test_da):
    layerregridder = imod.prepare.LayerRegridder(method="mode")
    source = test_da
    top_src = xr.full_like(source, 0.0)
    top_src.data[1, :, :] = -2.0
    bottom_src = xr.full_like(source, -2.0)
    bottom_src.data[1, :, :] = -4.0

    top_dst = xr.full_like(source, 0.0)
    top_dst.data[1, :, :] = -3.0
    bottom_dst = xr.full_like(source, -3.0)
    bottom_dst.data[1, :, :] = -4.0

    actual = layerregridder.regrid(source, top_src, bottom_src, top_dst, bottom_dst)
    coords = {"layer": [1, 2], "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    dims = ("layer", "y", "x")
    expected = xr.DataArray(
        [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]],
        coords,
        dims,
    )
    assert actual.identical(expected)


def test_layerregridder_topbot_nan():
    layerregridder = imod.prepare.LayerRegridder(method="mean")
    coords = {"x": [0.5], "y": [0.5], "layer": [1, 2, 3, 4]}
    dims = ("layer", "y", "x")
    shape = (4, 1, 1)
    source = xr.DataArray(np.ones(shape), coords, dims)
    top_src = xr.DataArray(
        np.array([np.nan, np.nan, 2.0, 1.0]).reshape(shape), coords, dims
    )
    bottom_src = xr.DataArray(
        np.array([np.nan, np.nan, 1.0, 0.0]).reshape(shape), coords, dims
    )
    top_dst = xr.DataArray(
        np.array([4.0, 3.0, 2.0, np.nan]).reshape(shape), coords, dims
    )
    bottom_dst = xr.DataArray(
        np.array([np.nan, 2.0, 1.0, np.nan]).reshape(shape), coords, dims
    )
    actual = layerregridder.regrid(source, top_src, bottom_src, top_dst, bottom_dst)

    expected = xr.DataArray(
        np.array([np.nan, np.nan, 1.0, np.nan]).reshape(shape), coords, dims
    )
    assert actual.identical(expected)
