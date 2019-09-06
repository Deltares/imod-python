import pytest
import numpy as np
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
    data = np.arange(np.product(shape), dtype=np.float).reshape(shape)
    source = xr.DataArray(data, coords, dims)
    return source


def test_voxelize__mean_1(test_da):
    voxelizer = imod.prepare.Voxelizer(method="mean")
    source = test_da
    z = [-0.5, -1.5, -2.5, -3.5]
    like = xr.DataArray(np.arange(4), {"z": z}, ["z"])

    top = xr.full_like(source, 0.0)
    top.data[1, :, :] = -2.0
    bottom = xr.full_like(source, -2.0)
    bottom.data[1, :, :] = -4.0

    actual = voxelizer.voxelize(source, top, bottom, like)
    coords = {"z": z, "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    coords["z"] = z
    dims = ("z", "y", "x")
    expected = xr.DataArray(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        coords,
        dims,
    )
    assert actual.identical(expected)


def test_voxelize__mean_2(test_da):
    voxelizer = imod.prepare.Voxelizer(method="mean")
    source = test_da
    z = [-0.5, -2.0, -3.5]
    dz = [-1.0, -2.0, -1.0]
    like = xr.DataArray(np.arange(3), {"z": z, "dz": ("z", dz)}, ["z"])

    top = xr.full_like(source, 0.0)
    top.data[1, :, :] = -2.0
    bottom = xr.full_like(source, -2.0)
    bottom.data[1, :, :] = -4.0

    actual = voxelizer.voxelize(source, top, bottom, like)
    coords = {"z": z, "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    coords["z"] = z
    dims = ("z", "y", "x")
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


def test_voxelize__max_overlap_1(test_da):
    voxelizer = imod.prepare.Voxelizer(method="max_overlap")
    source = test_da
    z = [-0.5, -2.5]
    dz = [-1.0, -3.0]
    like = xr.DataArray(np.arange(2), {"z": z, "dz": ("z", dz)}, ["z"])

    top = xr.full_like(source, 0.0)
    top.data[1, :, :] = -2.0
    bottom = xr.full_like(source, -2.0)
    bottom.data[1, :, :] = -4.0

    actual = voxelizer.voxelize(source, top, bottom, like)
    coords = {"z": z, "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    coords["z"] = z
    dims = ("z", "y", "x")
    expected = xr.DataArray(
        [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]],
        coords,
        dims,
    )
    assert actual.identical(expected)


def test_voxelize__max_overlap_2(test_da):
    voxelizer = imod.prepare.Voxelizer(method="max_overlap")
    source = test_da
    z = [-0.5, -2.5]
    dz = [-1.0, -3.0]
    like = xr.DataArray(np.arange(2), {"z": z, "dz": ("z", dz)}, ["z"])

    top = xr.full_like(source, -2.0)
    top.data[0, :, :] = 0.0
    top.data[0, 0, 0] = np.nan
    top.data[0, 1, 1] = np.nan
    bottom = xr.full_like(source, -4.0)
    bottom.data[0, :, :] = -2.0
    bottom.data[0, 0, 0] = np.nan
    bottom.data[0, 1, 1] = np.nan

    actual = voxelizer.voxelize(source, top, bottom, like)
    coords = {"z": z, "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}
    coords["z"] = z
    dims = ("z", "y", "x")
    expected = xr.DataArray(
        [
            [[np.nan, 1.0, 2.0], [3.0, np.nan, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        coords,
        dims,
    )
    assert actual.identical(expected)
