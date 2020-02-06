import os

import numba
import numpy as np
import pytest
import xarray as xr

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


def test_make_regrid():
    if "NUMBA_DISABLE_JIT" in os.environ:
        pass
    else:
        # Cannot really test functionality, since it's compiled by numba at runtime
        # This just checks whether it's ingested okay
        func = imod.prepare.regrid._jit_regrid(mean, 1)
        assert isinstance(func, numba.targets.registry.CPUDispatcher)

        func = imod.prepare.regrid._make_regrid(mean, 1)
        assert isinstance(func, numba.targets.registry.CPUDispatcher)


def test_regrid_1d():
    src_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    dst_x = np.array([0.0, 2.5, 5.0])
    alloc_len, i_w = imod.prepare.common._weights_1d(src_x, dst_x, False)
    inds_weights = [tuple(elem) for elem in i_w]
    values = np.zeros(alloc_len)
    weights = np.zeros(alloc_len)
    src = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    dst = np.array([0.0, 0.0])

    # Regrid method 1
    first_regrid = imod.prepare.regrid._jit_regrid(numba.njit(first), 1)
    dst = first_regrid(src, dst, values, weights, *inds_weights)
    assert np.allclose(dst, np.array([10.0, 30.0]))

    # Regrid method 2
    mean_regrid = imod.prepare.regrid._jit_regrid(numba.njit(mean), 1)
    dst = mean_regrid(src, dst, values, weights, *inds_weights)
    assert np.allclose(
        dst, np.array([(10.0 + 20.0 + 30.0) / 3.0, (30.0 + 40.0 + 50.0) / 3.0])
    )

    # Regrid method 3
    wmean_regrid = imod.prepare.regrid._jit_regrid(numba.njit(weightedmean), 1)
    dst = wmean_regrid(src, dst, values, weights, *inds_weights)
    assert np.allclose(
        dst,
        np.array([(10.0 + 20.0 + 0.5 * 30.0) / 2.5, (30.0 * 0.5 + 40.0 + 50.0) / 2.5]),
    )


def test_iter_regrid__1d():
    ndim_regrid = 1
    src_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    dst_x = np.array([0.0, 2.5, 5.0])
    alloc_len, i_w = imod.prepare.common._weights_1d(src_x, dst_x, False)
    inds_weights = [tuple(elem) for elem in i_w]

    # 1D regrid over 1D array
    src = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    dst = np.zeros(2)
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)
    iter_src, iter_dst = imod.prepare.common._reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, *inds_weights)
    assert np.allclose(dst, np.array([10.0, 30.0]))

    # 1D regrid over 2D array
    src = np.array([[10.0, 20.0, 30.0, 40.0, 50.0] for _ in range(3)])
    dst = np.zeros((3, 2))
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)
    iter_src, iter_dst = imod.prepare.common._reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, *inds_weights)
    assert np.allclose(dst, np.array([[10.0, 30.0], [10.0, 30.0], [10.0, 30.0]]))

    # 1D regrid over 3D array
    src = np.zeros((4, 3, 5))
    src[..., :] = [10.0, 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((4, 3, 2))
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)
    iter_src, iter_dst = imod.prepare.common._reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, *inds_weights)
    compare = np.zeros((4, 3, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_nd_regrid__1d():
    # 1D regrid over 3D array
    src_coords = (np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),)
    dst_coords = (np.array([0.0, 2.5, 5.0]),)
    ndim_regrid = len(src_coords)
    src = np.zeros((4, 3, 5))
    src[..., :] = [10.0, 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((4, 3, 2))
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)

    dst = imod.prepare.regrid._nd_regrid(
        src, dst, src_coords, dst_coords, iter_regrid, False
    )
    compare = np.zeros((4, 3, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_nd_regrid__2d__first():
    # 2D regrid over 3D array
    src_coords = (
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    dst_coords = (np.array([0.0, 2.5, 5.0]), np.array([0.0, 2.5, 5.0]))
    ndim_regrid = len(src_coords)
    src = np.zeros((4, 5, 5))
    src[..., :] = [10.0, 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((4, 2, 2))
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)

    dst = imod.prepare.regrid._nd_regrid(
        src, dst, src_coords, dst_coords, iter_regrid, False
    )
    compare = np.zeros((4, 2, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_nd_regrid__2d__mean():
    # 2D regrid over 3D array
    src_coords = (
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    dst_coords = (np.array([0.0, 2.5, 5.0]), np.array([0.0, 2.5, 5.0]))
    ndim_regrid = len(src_coords)
    src = np.zeros((4, 5, 5))
    src[..., :] = [10.0, 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((4, 2, 2))
    iter_regrid = imod.prepare.regrid._make_regrid(mean, ndim_regrid)

    dst = imod.prepare.regrid._nd_regrid(
        src, dst, src_coords, dst_coords, iter_regrid, False
    )
    compare = np.zeros((4, 2, 2))
    compare[..., :] = [20.0, 40.0]
    assert np.allclose(dst, compare)


def test_nd_regrid__3d__first():
    # 3D regrid over 3D array
    src_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    dst_x = np.array([0.0, 2.5, 5.0])
    src_coords = [src_x for _ in range(3)]
    dst_coords = [dst_x for _ in range(3)]
    ndim_regrid = len(src_coords)
    src = np.zeros((5, 5, 5))
    src[..., :] = [10.0, 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((2, 2, 2))
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)

    dst = imod.prepare.regrid._nd_regrid(
        src, dst, src_coords, dst_coords, iter_regrid, False
    )
    compare = np.zeros((2, 2, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_nd_regrid__4d3d__first():
    # 3D regrid over 4D array
    src_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    dst_x = np.array([0.0, 2.5, 5.0])
    src_coords = [src_x for _ in range(3)]
    dst_coords = [dst_x for _ in range(3)]
    ndim_regrid = len(src_coords)
    src = np.zeros((3, 5, 5, 5))
    src[..., :] = [10.0, 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((3, 2, 2, 2))
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)

    dst = imod.prepare.regrid._nd_regrid(
        src, dst, src_coords, dst_coords, iter_regrid, False
    )
    compare = np.zeros((3, 2, 2, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_regrid_coord():
    # Regular
    da = xr.DataArray((np.zeros(4)), {"x": np.arange(4.0) + 0.5}, ("x",))
    regridx = imod.prepare.common._coord(da, "x")
    assert np.allclose(regridx, np.arange(5.0))

    # Negative x
    da = xr.DataArray((np.zeros(4)), {"x": np.arange(-4.0, 0.0, 1.0) + 0.5}, ("x",))
    regridx = imod.prepare.common._coord(da, "x")
    assert np.allclose(regridx, np.arange(-4.0, 1.0, 1.0))

    # Negative dx
    da = xr.DataArray((np.zeros(4)), {"x": np.arange(0.0, -4.0, -1.0) - 0.5}, ("x",))
    regridx = imod.prepare.common._coord(da, "x")
    assert np.allclose(regridx, np.arange(0.0, -5.0, -1.0))

    # Non-equidistant, postive dx, negative dy
    nrow, ncol = 3, 4
    dx = np.array([0.9, 1.1, 0.8, 1.2])
    dy = np.array([-1.3, -0.7, -1.0])
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "nonequidistant", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)
    da = xr.DataArray(data, **kwargs)

    regridx = imod.prepare.common._coord(da, "x")
    regridy = imod.prepare.common._coord(da, "y")
    assert float(regridx.min()) == xmin
    assert float(regridx.max()) == xmax
    assert float(regridy.min()) == ymin
    assert float(regridy.max()) == ymax
    assert np.allclose(np.diff(regridx), dx)
    assert np.allclose(np.diff(regridy), dy)

    # Now test it if dy doesn't have the right sign
    # it should automatically infer it based on y instead.
    da["dy"].values *= -1.0
    regridy2 = imod.prepare.common._coord(da, "y")
    assert np.allclose(regridy, regridy2)


def test_regrid_mean1d():
    values = np.array([1.0, 2.0, 3.0])
    src_x = np.array([0.5, 1.5, 2.5])
    dst_x = np.array([0.5, 2.0])
    coords = {"x": src_x, "dx": ("x", np.array([1.0, 1.0, 1.0]))}
    like_coords = {"x": dst_x, "dx": ("x", np.array([1.0, 2.0]))}
    dims = ("x",)
    source = xr.DataArray(values, coords, dims)
    like = xr.DataArray(np.empty(2), like_coords, dims)
    out = imod.prepare.Regridder(method=weightedmean).regrid(source, like)
    compare = np.array([1.0, 2.5])
    assert np.allclose(out.values, compare)


def test_regrid_mean1d__dx_negative():
    values = np.array([1.0, 2.0, 3.0])
    src_x = np.array([2.5, 1.5, 0.5])
    dst_x = np.array([2.0, 0.5])
    coords = {"x": src_x, "dx": ("x", np.array([-1.0, -1.0, -1.0]))}
    like_coords = {"x": dst_x, "dx": ("x", np.array([-2.0, -1.0]))}
    dims = ("x",)
    source = xr.DataArray(values, coords, dims)
    like = xr.DataArray(np.empty(2), like_coords, dims)
    out = imod.prepare.Regridder(method=weightedmean).regrid(source, like)
    compare = np.array([1.5, 3.0])
    assert np.allclose(out.values, compare)


def test_regrid_mean2d():
    values = np.array([[0.6, 0.2, 3.4], [1.4, 1.6, 1.0], [4.0, 2.8, 3.0]])
    src_x = np.arange(3.0) + 0.5
    coords = {"y": src_x, "x": src_x}
    dims = ("y", "x")
    source = xr.DataArray(values, coords, dims)
    dst_x = np.arange(0.0, 3.0, 1.5) + 0.75
    likecoords = {"y": dst_x, "x": dst_x}
    like = xr.DataArray(np.empty((2, 2)), likecoords, dims)

    out = imod.prepare.Regridder(method=weightedmean).regrid(source, like)
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
    assert np.allclose(out.values, compare)


def test_regrid_mean2d_over3darray():
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

    out = imod.prepare.Regridder(method=weightedmean).regrid(source, like)
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

    assert np.allclose(out.values, compare)


def test_regrid_condutance2d():
    # First case, same domain, smaller cellsizes
    y = np.arange(10.0, 0.0, -2.5) - 1.25
    x = np.arange(0.0, 10.0, 2.5) + 1.25
    coords = {"y": y, "x": x}
    dims = ("y", "x")
    like_da = xr.DataArray(np.empty((4, 4)), coords, dims)
    src_da = xr.DataArray(
        [[10.0, 10.0], [10.0, 10.0]], {"y": [7.5, 2.5], "x": [2.5, 7.5]}, dims
    )

    regridder = imod.prepare.Regridder(method=conductance, use_relative_weights=True)
    dst_da = regridder.regrid(src_da, like_da)
    assert float(src_da.sum()) == float(dst_da.sum())

    # Second case, different domain, smaller cellsizes
    dx = np.array([2.5, 2.5, 2.5, 3.5])
    x = np.cumsum(dx) - dx * 0.5
    coords["x"] = x
    coords["dx"] = ("x", dx)
    like_da = xr.DataArray(np.empty((4, 4)), coords, dims)
    dst_da = regridder.regrid(src_da, like_da)
    assert float(src_da.sum()) == float(dst_da.sum())

    # Third case, same domain, small to large cellsizes
    y = np.arange(10.0, 0.0, -2.5) - 1.25
    x = np.arange(0.0, 10.0, 2.5) + 1.25
    coords = {"y": y, "x": x}
    dims = ("y", "x")
    src_da = xr.DataArray(np.full((4, 4), 10.0), coords, dims)
    like_da = xr.DataArray(
        [[10.0, 10.0], [10.0, 10.0]], {"y": [7.5, 2.5], "x": [2.5, 7.5]}, dims
    )

    dst_da = regridder.regrid(src_da, like_da)
    assert float(src_da.sum()) == float(dst_da.sum())

    # Fourth case, larger domain, small to large cellsizes
    like_da = xr.DataArray(
        [[10.0, 10.0], [10.0, 10.0]], {"y": [15.0, 5.0], "x": [5.0, 15.0]}, dims
    )

    dst_da = regridder.regrid(src_da, like_da)
    assert float(src_da.sum()) == float(dst_da.sum())


def test_regrid_conductance3d__errors():
    values = np.array([[0.6, 0.2, 3.4], [1.4, 1.6, 1.0], [4.0, 2.8, 3.0]])
    values = np.stack([values for _ in range(5)])
    src_x = np.arange(3.0) + 0.5
    src_z = np.arange(5.0)
    coords = {"z": src_z, "y": src_x, "x": src_x}
    dims = ("z", "y", "x")
    source = xr.DataArray(values, coords, dims)
    dst_x = np.arange(0.0, 3.0, 1.5) + 0.75
    dst_z = np.arange(0.0, 2.5, 0.5)
    likecoords = {"z": dst_z, "y": dst_x, "x": dst_x}
    like = xr.DataArray(np.empty((5, 2, 2)), likecoords, dims)

    with pytest.raises(ValueError):
        _ = imod.prepare.Regridder(method="conductance").regrid(source, like)


def test_str_method():
    values = np.array([1.0, 2.0, 3.0])
    src_x = np.array([2.5, 1.5, 0.5])
    dst_x = np.array([2.0, 0.5])
    coords = {"x": src_x, "dx": ("x", np.array([-1.0, -1.0, -1.0]))}
    like_coords = {"x": dst_x, "dx": ("x", np.array([-2.0, -1.0]))}
    dims = ("x",)
    source = xr.DataArray(values, coords, dims)
    like = xr.DataArray(np.empty(2), like_coords, dims)
    # Test function method
    out = imod.prepare.Regridder(method=mean).regrid(source, like)
    compare = np.array([1.5, 3.0])
    assert np.allclose(out.values, compare)

    # Now test str method
    out = imod.prepare.Regridder(method="mean").regrid(source, like)
    assert np.allclose(out.values, compare)

    out = imod.prepare.Regridder(method="nearest").regrid(source, like)
