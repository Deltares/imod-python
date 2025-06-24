"""
Test if xugrid regridding methods work as expected, and return the same reults
as the now removed imod python Regridder.
"""

import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod


def first(values, weights):
    return values[0]


def mean(values, weights):
    vsum = 0.0
    wsum = 0.0
    for i in range(values.size):
        v = values[i]
        vsum += v
        wsum += 1.0
    return vsum / wsum


def weightedmean(values, weights):
    vsum = 0.0
    wsum = 0.0
    for i in range(values.size):
        v = values[i]
        w = weights[i]
        vsum += w * v
        wsum += w
    return vsum / wsum


def weightedmean_xu(values, weights, workspace):
    vsum = 0.0
    wsum = 0.0
    for i in range(values.size):
        v = values[i]
        w = weights[i]
        vsum += w * v
        wsum += w
    return vsum / wsum


def conductance(values, weights):
    v_agg = 0.0
    w_sum = 0.0
    for i in range(values.size):
        v = values[i]
        w = weights[i]
        if np.isnan(v):
            continue
        v_agg += v * w
        w_sum += w
    if w_sum == 0:
        return np.nan
    else:
        return v_agg


@pytest.mark.parametrize("chunksize", [1, 2, 3])
def test_regrid_mean2d(chunksize):
    values = np.array([[0.6, 0.2, 3.4], [1.4, 1.6, 1.0], [4.0, 2.8, 3.0]])
    src_x = np.arange(3.0) + 0.5
    coords = {"y": src_x, "x": src_x}
    dims = ("y", "x")
    source = xr.DataArray(values, coords, dims)
    dst_x = np.arange(0.0, 3.0, 1.5) + 0.75
    likecoords = {"y": dst_x, "x": dst_x}
    like = xr.DataArray(np.empty((2, 2)), likecoords, dims)

    compare = np.array(
        [
            [
                (0.6 + 0.5 * 0.2 + 0.5 * 1.4 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
                (3.4 + 0.5 * 0.2 + 0.5 * 1.0 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
            ],
            [
                (4.0 + 0.5 * 1.4 + 0.5 * 2.8 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
                (3.0 + 0.5 * 1.0 + 0.5 * 2.8 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
            ],
        ]
    )
    # Verify xugrid returns same result
    out_xu = xu.OverlapRegridder(
        source=source, target=like, method=weightedmean_xu
    ).regrid(source)
    assert np.allclose(out_xu.values, compare)

    # Now with chunking
    source = source.chunk({"x": chunksize, "y": chunksize})
    # Verify xugrid returns same result
    out_xu = xu.OverlapRegridder(
        source=source, target=like, method=weightedmean_xu
    ).regrid(source)
    assert np.allclose(out_xu.values, compare)


@pytest.mark.parametrize("chunksize", [1, 2, 3])
def test_regrid_mean2d_over3darray(chunksize):
    values = np.array([[0.6, 0.2, 3.4], [1.4, 1.6, 1.0], [4.0, 2.8, 3.0]])
    values = np.stack([values for _ in range(5)])
    src_x = np.arange(3.0) + 0.5
    src_z = np.arange(5.0)
    coords = {"z": src_z, "y": src_x, "x": src_x}
    dims = ("z", "y", "x")
    source = xr.DataArray(values, coords, dims)
    dst_x = np.arange(0.0, 3.0, 1.5) + 0.75
    likecoords = {"z": src_z, "y": dst_x, "x": dst_x}
    like = xr.DataArray(np.empty((5, 2, 2)), likecoords, dims)

    compare_values = np.array(
        [
            [
                (0.6 + 0.5 * 0.2 + 0.5 * 1.4 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
                (3.4 + 0.5 * 0.2 + 0.5 * 1.0 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
            ],
            [
                (4.0 + 0.5 * 1.4 + 0.5 * 2.8 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
                (3.0 + 0.5 * 1.0 + 0.5 * 2.8 + 0.25 * 1.6) / (1.0 + 0.5 + 0.5 + 0.25),
            ],
        ]
    )
    compare = np.empty((5, 2, 2))
    compare[:, ...] = compare_values
    # Verify xugrid returns same result
    out_xu = xu.OverlapRegridder(
        source=source, target=like, method=weightedmean_xu
    ).regrid(source)
    assert np.allclose(out_xu.values, compare)

    # Now with chunking
    source = source.chunk({"x": chunksize, "y": chunksize})
    # Verify xugrid returns same result
    out_xu = xu.OverlapRegridder(
        source=source, target=like, method=weightedmean_xu
    ).regrid(source)
    assert np.allclose(out_xu.values, compare)


def test_regrid_conductance2d():
    # First case, same domain, smaller cellsizes
    y = np.arange(10.0, 0.0, -2.5) - 1.25
    x = np.arange(0.0, 10.0, 2.5) + 1.25
    coords = {"y": y, "x": x}
    dims = ("y", "x")
    like_da = xr.DataArray(np.empty((4, 4)), coords, dims)
    src_da = xr.DataArray(
        [[10.0, 10.0], [10.0, 10.0]], {"y": [7.5, 2.5], "x": [2.5, 7.5]}, dims
    )
    src_da[0, 0] = np.nan

    # Verify xugrid returns same result
    dst_da_xu = xu.RelativeOverlapRegridder(
        source=src_da, target=like_da, method="conductance"
    ).regrid(src_da)
    assert float(src_da.sum()) == float(dst_da_xu.sum())

    # Second case, different domain, smaller cellsizes
    dx = np.array([2.5, 2.5, 2.5, 3.5])
    x = np.cumsum(dx) - dx * 0.5
    coords["x"] = x
    coords["dx"] = ("x", dx)
    like_da = xr.DataArray(np.empty((4, 4)), coords, dims)
    # Verify xugrid returns same result
    dst_da_xu = xu.RelativeOverlapRegridder(
        source=src_da, target=like_da, method="conductance"
    ).regrid(src_da)
    assert float(src_da.sum()) == float(dst_da_xu.sum())

    # Third case, same domain, small to large cellsizes
    y = np.arange(10.0, 0.0, -2.5) - 1.25
    x = np.arange(0.0, 10.0, 2.5) + 1.25
    coords = {"y": y, "x": x}
    dims = ("y", "x")
    src_da = xr.DataArray(np.full((4, 4), 10.0), coords, dims)
    src_da[0, 0] = np.nan
    like_da = xr.DataArray(
        [[10.0, 10.0], [10.0, 10.0]], {"y": [7.5, 2.5], "x": [2.5, 7.5]}, dims
    )
    # Verify xugrid returns same result
    dst_da_xu = xu.RelativeOverlapRegridder(
        source=src_da, target=like_da, method="conductance"
    ).regrid(src_da)
    assert float(src_da.sum()) == float(dst_da_xu.sum())

    # Fourth case, larger domain, small to large cellsizes
    like_da = xr.DataArray(
        [[10.0, 10.0], [10.0, 10.0]], {"y": [15.0, 5.0], "x": [5.0, 15.0]}, dims
    )
    # Verify xugrid returns same result
    dst_da_xu = xu.RelativeOverlapRegridder(
        source=src_da, target=like_da, method="conductance"
    ).regrid(src_da)
    assert float(src_da.sum()) == float(dst_da_xu.sum())


def test_notimplementederror_upon_init():
    with pytest.raises(NotImplementedError):
        _ = imod.prepare.Regridder(method="mean")
