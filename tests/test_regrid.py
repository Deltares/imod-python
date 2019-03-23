import numba
import numpy as np
import pytest
import xarray as xr
from imod import regrid


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
    assert regrid._overlap((0., 1.), (0., 2.)) == 1.
    assert regrid._overlap((-1., 1.), (0., 2.)) == 1.
    assert regrid._overlap((-1., 3.), (0., 2.)) == 2.
    assert regrid._overlap((-1., 3.), (-2., 2.)) == 3.


def test_starts():
    # Complete range
    src_x = np.arange(0., 11., 1.)
    dst_x = np.arange(0., 11., 2.5)
    # Returns tuples with (src_ind, dst_ind)
    assert list(regrid._starts(src_x, dst_x)) == [(0, 0), (1, 2), (2, 5), (3, 7)]

    # Partial dst
    dst_x = np.arange(5., 11., 2.5)
    assert list(regrid._starts(src_x, dst_x)) == [(0, 5), (1, 7)]

    # Partial src
    src_x = np.arange(5., 11., 1.)
    dst_x = np.arange(0., 11., 2.5)
    assert list(regrid._starts(src_x, dst_x)) == [(0, 0), (1, 0), (2, 0), (3, 2)]

    # Irregular grid
    src_x = np.array([0., 2.5, 7.5, 10.0])
    dst_x = np.array([0., 5., 10.])
    assert list(regrid._starts(src_x, dst_x)) == [(0, 0), (1, 1)]

    # Negative coords
    src_x = np.arange(-20., -9., 1.)
    dst_x = np.arange(-20., -9., 2.5)
    assert list(regrid._starts(src_x, dst_x)) == [(0, 0), (1, 2), (2, 5), (3, 7)]

    # Mixed coords
    src_x = np.arange(-5., 6., 1.)
    dst_x = np.arange(-5., 6., 2.5)
    assert list(regrid._starts(src_x, dst_x)) == [(0, 0), (1, 2), (2, 5), (3, 7)]


def test_weights():
    src_x = np.arange(0., 11., 1.)
    dst_x = np.arange(0., 11., 2.5)
    max_len, (dst_inds, src_inds, weights) = regrid._weights_1d(src_x, dst_x)
    assert max_len == 3
    assert dst_inds == [0, 1, 2, 3]
    assert src_inds == [[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 9]]
    assert weights == [[1., 1., 0.5], [0.5, 1., 1.], [1., 1., 0.5], [0.5, 1., 1.]]

    # Irregular grid
    src_x = np.array([0., 2.5, 7.5, 10.0])
    dst_x = np.array([0., 5., 10.])
    max_len, (dst_inds, src_inds, weights) = regrid._weights_1d(src_x, dst_x)
    assert max_len == 2
    assert dst_inds == [0, 1]
    assert src_inds == [[0, 1], [1, 2]]
    assert weights == [[2.5, 2.5], [2.5, 2.5]]

    # Mixed coords
    src_x = np.arange(-5., 6., 1.)
    dst_x = np.arange(-5., 6., 2.5)
    max_len, (dst_inds, src_inds, weights) = regrid._weights_1d(src_x, dst_x)
    assert max_len == 3
    assert dst_inds == [0, 1, 2, 3]
    assert src_inds == [[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 9]]
    assert weights == [[1., 1., 0.5], [0.5, 1., 1.], [1., 1., 0.5], [0.5, 1., 1.]]


def test_reshape():
    src = np.zeros((3, 5))
    dst = np.zeros((3, 2))
    iter_src, iter_dst = regrid._reshape(src, dst, ndim_regrid=1)
    assert iter_src.shape == (3, 5)
    assert iter_dst.shape == (3, 2)

    src = np.zeros((2, 4, 3, 5))
    dst = np.zeros((2, 4, 3, 2))
    iter_src, iter_dst = regrid._reshape(src, dst, ndim_regrid=1)
    assert iter_src.shape == (24, 5)
    assert iter_dst.shape == (24, 2)

    src = np.zeros((3, 5))
    dst = np.zeros((3, 2))
    iter_src, iter_dst = regrid._reshape(src, dst, ndim_regrid=2)
    assert iter_src.shape == (1, 3, 5)
    assert iter_dst.shape == (1, 3, 2)

    src = np.zeros((2, 4, 3, 5))
    dst = np.zeros((2, 4, 3, 2))
    iter_src, iter_dst = regrid._reshape(src, dst, ndim_regrid=3)
    assert iter_src.shape == (2, 4, 3, 5)
    assert iter_dst.shape == (2, 4, 3, 2)


def test_make_regrid():
    # Cannot really test functionality, since it's compiled by numba at runtime
    # This just checks whether it's ingested okay
    func = regrid._jit_regrid(mean, 1)
    assert isinstance(func, numba.targets.registry.CPUDispatcher)

    func = regrid._make_regrid(mean, 1)
    assert isinstance(func, numba.targets.registry.CPUDispatcher)


def test_regrid_1d():
    src_x = np.array([0., 1., 2., 3., 4., 5.])
    dst_x = np.array([0.0, 2.5, 5.0])
    alloc_len, i_w = regrid._weights_1d(src_x, dst_x)
    inds_weights = [tuple(tuple(elem) for elem in i_w)]
    values = np.zeros(alloc_len)
    weights = np.zeros(alloc_len)
    src = np.array([10., 20.0, 30.0, 40.0, 50.0])
    dst = np.array([0.0, 0.0])

    # Regrid method 1
    first_regrid = regrid._jit_regrid(numba.njit(first), 1)
    dst = first_regrid(src, dst, values, weights, inds_weights)
    assert np.allclose(dst, np.array([10.0, 30.0]))

    # Regrid method 2
    mean_regrid = regrid._jit_regrid(numba.njit(mean), 1)
    dst = mean_regrid(src, dst, values, weights, inds_weights)
    assert np.allclose(dst, np.array([(10. + 20. + 30.) / 3., (30. + 40. + 50.) / 3.]))

    # Regrid method 3
    wmean_regrid = regrid._jit_regrid(numba.njit(weightedmean), 1)
    dst = wmean_regrid(src, dst, values, weights, inds_weights)
    assert np.allclose(
        dst, np.array([(10. + 20. + 0.5 * 30.) / 2.5, (30. * 0.5 + 40. + 50.) / 2.5])
    )


def test_iter_regrid__1d():
    ndim_regrid = 1
    src_x = np.array([0., 1., 2., 3., 4., 5.])
    dst_x = np.array([0.0, 2.5, 5.0])
    alloc_len, i_w = regrid._weights_1d(src_x, dst_x)
    inds_weights = [tuple(tuple(elem) for elem in i_w)]

    # 1D regrid over 1D array
    src = np.array([10., 20.0, 30.0, 40.0, 50.0])
    dst = np.zeros(2)
    iter_regrid = regrid._make_regrid(first, ndim_regrid)
    iter_src, iter_dst = regrid._reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, inds_weights)
    assert np.allclose(dst, np.array([10.0, 30.0]))

    # 1D regrid over 2D array
    src = np.array([[10., 20.0, 30.0, 40.0, 50.0] for _ in range(3)])
    dst = np.zeros((3, 2))
    iter_regrid = regrid._make_regrid(first, ndim_regrid)
    iter_src, iter_dst = regrid._reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, inds_weights)
    assert np.allclose(dst, np.array([[10.0, 30.0], [10.0, 30.0], [10.0, 30.0]]))

    # 1D regrid over 3D array
    src = np.zeros((4, 3, 5))
    src[..., :] = [10., 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((4, 3, 2))
    iter_regrid = regrid._make_regrid(first, ndim_regrid)
    iter_src, iter_dst = regrid._reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, inds_weights)
    compare = np.zeros((4, 3, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_nd_regrid__1d():
    # 1D regrid over 3D array
    src_coords = (np.array([0., 1., 2., 3., 4., 5.]),)
    dst_coords = (np.array([0.0, 2.5, 5.0]),)
    ndim_regrid = len(src_coords)
    src = np.zeros((4, 3, 5))
    src[..., :] = [10., 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((4, 3, 2))
    iter_regrid = regrid._make_regrid(first, ndim_regrid)

    dst = regrid._nd_regrid(src, dst, src_coords, dst_coords, iter_regrid)
    compare = np.zeros((4, 3, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_nd_regrid__2d__first():
    # 2D regrid over 3D array
    src_coords = (np.array([0., 1., 2., 3., 4., 5.]), np.array([0., 1., 2., 3., 4., 5.]))
    dst_coords = (np.array([0.0, 2.5, 5.0]), np.array([0.0, 2.5, 5.0]))
    ndim_regrid = len(src_coords)
    src = np.zeros((4, 5, 5))
    src[..., :] = [10., 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((4, 2, 2))
    iter_regrid = regrid._make_regrid(first, ndim_regrid)

    dst = regrid._nd_regrid(src, dst, src_coords, dst_coords, iter_regrid)
    compare = np.zeros((4, 2, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_nd_regrid__2d__mean():
    # 2D regrid over 3D array
    src_coords = (np.array([0., 1., 2., 3., 4., 5.]), np.array([0., 1., 2., 3., 4., 5.]))
    dst_coords = (np.array([0.0, 2.5, 5.0]), np.array([0.0, 2.5, 5.0]))
    ndim_regrid = len(src_coords)
    src = np.zeros((4, 5, 5))
    src[..., :] = [10., 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((4, 2, 2))
    iter_regrid = regrid._make_regrid(mean, ndim_regrid)

    dst = regrid._nd_regrid(src, dst, src_coords, dst_coords, iter_regrid)
    compare = np.zeros((4, 2, 2))
    compare[..., :] = [20.0, 40.0]
    assert np.allclose(dst, compare)


def test_nd_regrid__3d__first():
    # 3D regrid over 3D array
    src_x = np.array([0., 1., 2., 3., 4., 5.])
    dst_x = np.array([0.0, 2.5, 5.0])
    src_coords = [src_x for _ in range(3)]
    dst_coords = [dst_x for _ in range(3)]
    ndim_regrid = len(src_coords)
    src = np.zeros((5, 5, 5))
    src[..., :] = [10., 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((2, 2, 2))
    iter_regrid = regrid._make_regrid(first, ndim_regrid)

    dst = regrid._nd_regrid(src, dst, src_coords, dst_coords, iter_regrid)
    compare = np.zeros((2, 2, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)


def test_nd_regrid__4d3d__first():
    # 3D regrid over 4D array
    src_x = np.array([0., 1., 2., 3., 4., 5.])
    dst_x = np.array([0.0, 2.5, 5.0])
    src_coords = [src_x for _ in range(3)]
    dst_coords = [dst_x for _ in range(3)]
    ndim_regrid = len(src_coords)
    src = np.zeros((3, 5, 5, 5))
    src[..., :] = [10., 20.0, 30.0, 40.0, 50.0]
    dst = np.zeros((3, 2, 2, 2))
    iter_regrid = regrid._make_regrid(first, ndim_regrid)

    dst = regrid._nd_regrid(src, dst, src_coords, dst_coords, iter_regrid)
    compare = np.zeros((3, 2, 2, 2))
    compare[..., :] = [10.0, 30.0]
    assert np.allclose(dst, compare)