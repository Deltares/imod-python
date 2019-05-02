import numba
import numpy as np
import pytest
import xarray as xr
import sys

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


def test_overlap():
    assert imod.prepare.regrid._overlap((0.0, 1.0), (0.0, 2.0)) == 1.0
    assert imod.prepare.regrid._overlap((-1.0, 1.0), (0.0, 2.0)) == 1.0
    assert imod.prepare.regrid._overlap((-1.0, 3.0), (0.0, 2.0)) == 2.0
    assert imod.prepare.regrid._overlap((-1.0, 3.0), (-2.0, 2.0)) == 3.0


def test_starts():
    @numba.njit
    def get_starts(src_x, dst_x):
        result = []
        for i, j in imod.prepare.regrid._starts(src_x, dst_x):
            result.append((i, j))
        return result

    # Complete range
    src_x = np.arange(0.0, 11.0, 1.0)
    dst_x = np.arange(0.0, 11.0, 2.5)
    # Returns tuples with (src_ind, dst_ind)
    # List comprehension gives PicklingError
    result = get_starts(src_x, dst_x)
    assert result == [(0, 0), (1, 2), (2, 5), (3, 7)]

    # Partial dst
    dst_x = np.arange(5.0, 11.0, 2.5)
    result = get_starts(src_x, dst_x)
    assert result == [(0, 5), (1, 7)]

    # Partial src
    src_x = np.arange(5.0, 11.0, 1.0)
    dst_x = np.arange(0.0, 11.0, 2.5)
    result = get_starts(src_x, dst_x)
    assert result == [(0, 0), (1, 0), (2, 0), (3, 2)]

    # Irregular grid
    src_x = np.array([0.0, 2.5, 7.5, 10.0])
    dst_x = np.array([0.0, 5.0, 10.0])
    result = get_starts(src_x, dst_x)
    assert result == [(0, 0), (1, 1)]

    # Negative coords
    src_x = np.arange(-20.0, -9.0, 1.0)
    dst_x = np.arange(-20.0, -9.0, 2.5)
    result = get_starts(src_x, dst_x)
    assert result == [(0, 0), (1, 2), (2, 5), (3, 7)]

    # Mixed coords
    src_x = np.arange(-5.0, 6.0, 1.0)
    dst_x = np.arange(-5.0, 6.0, 2.5)
    result = get_starts(src_x, dst_x)
    assert result == [(0, 0), (1, 2), (2, 5), (3, 7)]


def test_weights():
    src_x = np.arange(0.0, 11.0, 1.0)
    dst_x = np.arange(0.0, 11.0, 2.5)
    max_len, (dst_inds, src_inds, weights) = imod.prepare.regrid._weights_1d(
        src_x, dst_x
    )
    assert max_len == 3
    assert np.allclose(dst_inds, np.array([0, 1, 2, 3]))
    assert np.allclose(src_inds, np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 9]]))
    assert np.allclose(
        weights,
        np.array([[1.0, 1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, 0.5], [0.5, 1.0, 1.0]]),
    )

    # Irregular grid
    src_x = np.array([0.0, 2.5, 7.5, 10.0])
    dst_x = np.array([0.0, 5.0, 10.0])
    max_len, (dst_inds, src_inds, weights) = imod.prepare.regrid._weights_1d(
        src_x, dst_x
    )
    assert max_len == 2
    assert np.allclose(dst_inds, np.array([0, 1]))
    assert np.allclose(src_inds, np.array([[0, 1], [1, 2]]))
    assert np.allclose(weights, np.array([[2.5, 2.5], [2.5, 2.5]]))

    # Mixed coords
    src_x = np.arange(-5.0, 6.0, 1.0)
    dst_x = np.arange(-5.0, 6.0, 2.5)
    max_len, (dst_inds, src_inds, weights) = imod.prepare.regrid._weights_1d(
        src_x, dst_x
    )
    assert max_len == 3
    assert np.allclose(dst_inds, np.array([0, 1, 2, 3]))
    assert np.allclose(src_inds, np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 9]]))
    assert np.allclose(
        weights,
        np.array([[1.0, 1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, 0.5], [0.5, 1.0, 1.0]]),
    )


def test_reshape():
    src = np.zeros((3, 5))
    dst = np.zeros((3, 2))
    iter_src, iter_dst = imod.prepare.regrid._reshape(src, dst, ndim_regrid=1)
    assert iter_src.shape == (3, 5)
    assert iter_dst.shape == (3, 2)

    src = np.zeros((2, 4, 3, 5))
    dst = np.zeros((2, 4, 3, 2))
    iter_src, iter_dst = imod.prepare.regrid._reshape(src, dst, ndim_regrid=1)
    assert iter_src.shape == (24, 5)
    assert iter_dst.shape == (24, 2)

    src = np.zeros((3, 5))
    dst = np.zeros((3, 2))
    iter_src, iter_dst = imod.prepare.regrid._reshape(src, dst, ndim_regrid=2)
    assert iter_src.shape == (1, 3, 5)
    assert iter_dst.shape == (1, 3, 2)

    src = np.zeros((2, 4, 3, 5))
    dst = np.zeros((2, 4, 3, 2))
    iter_src, iter_dst = imod.prepare.regrid._reshape(src, dst, ndim_regrid=3)
    assert iter_src.shape == (2, 4, 3, 5)
    assert iter_dst.shape == (2, 4, 3, 2)


def test_make_regrid():
    # Cannot really test functionality, since it's compiled by numba at runtime
    # This just checks whether it's ingested okay
    func = imod.prepare.regrid._jit_regrid(mean, 1)
    assert isinstance(func, numba.targets.registry.CPUDispatcher)

    func = imod.prepare.regrid._make_regrid(mean, 1)
    assert isinstance(func, numba.targets.registry.CPUDispatcher)


def test_regrid_1d():
    src_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    dst_x = np.array([0.0, 2.5, 5.0])
    alloc_len, i_w = imod.prepare.regrid._weights_1d(src_x, dst_x)
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
    alloc_len, i_w = imod.prepare.regrid._weights_1d(src_x, dst_x)
    inds_weights = [tuple(elem) for elem in i_w]

    # 1D regrid over 1D array
    src = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    dst = np.zeros(2)
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)
    iter_src, iter_dst = imod.prepare.regrid._reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, *inds_weights)
    assert np.allclose(dst, np.array([10.0, 30.0]))

    # 1D regrid over 2D array
    src = np.array([[10.0, 20.0, 30.0, 40.0, 50.0] for _ in range(3)])
    dst = np.zeros((3, 2))
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)
    iter_src, iter_dst = imod.prepare.regrid._reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, *inds_weights)
    assert np.allclose(dst, np.array([[10.0, 30.0], [10.0, 30.0], [10.0, 30.0]]))

    # 1D regrid over 3D array
    src = np.zeros((4, 3, 5))
    src[..., :] = [10.0, 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((4, 3, 2))
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)
    iter_src, iter_dst = imod.prepare.regrid._reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, *inds_weights)
    compare = np.zeros((4, 3, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_strictly_increasing():
    src_x = np.arange(5.0)
    dst_x = np.arange(5.0)
    _src_x, _dst_x = imod.prepare.regrid._strictly_increasing(src_x, dst_x)
    assert np.allclose(src_x, _src_x)
    assert np.allclose(dst_x, _dst_x)

    src_x = np.arange(5.0, 0.0, -1.0)
    dst_x = np.arange(5.0, 0.0, -1.0)
    _src_x, _dst_x = imod.prepare.regrid._strictly_increasing(src_x, dst_x)
    assert np.allclose(src_x[::-1], _src_x)
    assert np.allclose(dst_x[::-1], _dst_x)

    src_x = np.arange(5.0, 0.0, -1.0)
    dst_x = np.arange(5.0)
    with pytest.raises(ValueError):
        _src_x, _dst_x = imod.prepare.regrid._strictly_increasing(src_x, dst_x)


def test_nd_regrid__1d():
    # 1D regrid over 3D array
    src_coords = (np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),)
    dst_coords = (np.array([0.0, 2.5, 5.0]),)
    ndim_regrid = len(src_coords)
    src = np.zeros((4, 3, 5))
    src[..., :] = [10.0, 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((4, 3, 2))
    iter_regrid = imod.prepare.regrid._make_regrid(first, ndim_regrid)

    dst = imod.prepare.regrid._nd_regrid(src, dst, src_coords, dst_coords, iter_regrid)
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

    dst = imod.prepare.regrid._nd_regrid(src, dst, src_coords, dst_coords, iter_regrid)
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

    dst = imod.prepare.regrid._nd_regrid(src, dst, src_coords, dst_coords, iter_regrid)
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

    dst = imod.prepare.regrid._nd_regrid(src, dst, src_coords, dst_coords, iter_regrid)
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

    dst = imod.prepare.regrid._nd_regrid(src, dst, src_coords, dst_coords, iter_regrid)
    compare = np.zeros((3, 2, 2, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_regrid_coord():
    # Regular
    da = xr.DataArray((np.zeros(4)), {"x": np.arange(4.0) + 0.5}, ("x",))
    regridx = imod.prepare.regrid._coord(da, "x")
    assert np.allclose(regridx, np.arange(5.0))

    # Negative x
    da = xr.DataArray((np.zeros(4)), {"x": np.arange(-4.0, 0.0, 1.0) + 0.5}, ("x",))
    regridx = imod.prepare.regrid._coord(da, "x")
    assert np.allclose(regridx, np.arange(-4.0, 1.0, 1.0))

    # Negative dx
    da = xr.DataArray((np.zeros(4)), {"x": np.arange(0.0, -4.0, -1.0) - 0.5}, ("x",))
    regridx = imod.prepare.regrid._coord(da, "x")
    assert np.allclose(regridx, np.arange(0.0, -5.0, -1.0))

    # Non-equidistant, postive dx, negative dy
    nrow, ncol = 3, 4
    dx = np.array([0.9, 1.1, 0.8, 1.2])
    dy = np.array([-1.3, -0.7, -1.0])
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = imod.idf._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "nonequidistant", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)
    da = xr.DataArray(data, **kwargs)

    regridx = imod.prepare.regrid._coord(da, "x")
    regridy = imod.prepare.regrid._coord(da, "y")
    assert float(regridx.min()) == xmin
    assert float(regridx.max()) == xmax
    assert float(regridy.min()) == ymin
    assert float(regridy.max()) == ymax
    assert np.allclose(np.diff(regridx), dx)
    assert np.allclose(np.diff(regridy), dy)


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


# TODO: test nan values
# Implement different methods: ignore, or accept
