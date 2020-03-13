import os
import warnings

import numba
import numpy as np
import pytest
import xarray as xr

import imod


def test_starts():
    @numba.njit
    def get_starts(src_x, dst_x):
        result = []
        for i, j in imod.prepare.common._starts(src_x, dst_x):
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
    max_len, (dst_inds, src_inds, weights) = imod.prepare.common._weights_1d(
        src_x, dst_x, False
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
    max_len, (dst_inds, src_inds, weights) = imod.prepare.common._weights_1d(
        src_x, dst_x, False
    )
    assert max_len == 2
    assert np.allclose(dst_inds, np.array([0, 1]))
    assert np.allclose(src_inds, np.array([[0, 1], [1, 2]]))
    assert np.allclose(weights, np.array([[2.5, 2.5], [2.5, 2.5]]))

    # Mixed coords
    src_x = np.arange(-5.0, 6.0, 1.0)
    dst_x = np.arange(-5.0, 6.0, 2.5)
    max_len, (dst_inds, src_inds, weights) = imod.prepare.common._weights_1d(
        src_x, dst_x, False
    )
    assert max_len == 3
    assert np.allclose(dst_inds, np.array([0, 1, 2, 3]))
    assert np.allclose(src_inds, np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 9]]))
    assert np.allclose(
        weights,
        np.array([[1.0, 1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, 0.5], [0.5, 1.0, 1.0]]),
    )


def test_relative_weights():
    # In the test above, the absolute weights are the same as the relative weights
    # To have a test case, we simply multiply coordinates by two, while the
    # relative weights should remain the same.
    src_x = np.arange(0.0, 11.0, 1.0) * 2.0
    dst_x = np.arange(0.0, 11.0, 2.5) * 2.0
    max_len, (dst_inds, src_inds, weights) = imod.prepare.common._weights_1d(
        src_x, dst_x, True
    )
    assert max_len == 3
    assert np.allclose(dst_inds, np.array([0, 1, 2, 3]))
    assert np.allclose(src_inds, np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 9]]))
    assert np.allclose(
        weights,
        np.array([[1.0, 1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, 0.5], [0.5, 1.0, 1.0]]),
    )

    # Something non-equidistant
    src_x = np.array([0.0, 1.5])
    dst_x = np.array([0.0, 3.0])
    max_len, (dst_inds, src_inds, weights) = imod.prepare.common._weights_1d(
        src_x, dst_x, True
    )
    assert np.allclose(weights, np.array([[1.0]]))

    src_x = np.array([0.0, 3.0])
    dst_x = np.array([0.0, 1.5])
    max_len, (dst_inds, src_inds, weights) = imod.prepare.common._weights_1d(
        src_x, dst_x, True
    )
    assert np.allclose(weights, np.array([[0.5]]))


def test_area_weighted_methods():
    values = np.arange(5.0)
    weights = np.arange(0.0, 50.0, 10.0)
    values[0] = np.nan

    assert np.allclose(imod.prepare.common.mean(values, weights), 3.0)
    assert np.allclose(imod.prepare.common.harmonic_mean(values, weights), 2.5)
    assert np.allclose(
        imod.prepare.common.geometric_mean(values, weights), 2.780778340631819
    )

    values[1] = 3.0
    assert np.allclose(imod.prepare.common.mode(values, weights), 3.0)

    # Check if no issues arise with all nan
    values[:] = np.nan
    assert np.isnan(imod.prepare.common.mean(values, weights))
    assert np.isnan(imod.prepare.common.harmonic_mean(values, weights))
    assert np.isnan(imod.prepare.common.geometric_mean(values, weights))
    assert np.isnan(imod.prepare.common.mode(values, weights))


def test_methods():
    values = np.arange(5.0)
    weights = np.arange(0.0, 50.0, 10.0)
    values[0] = np.nan
    assert np.allclose(imod.prepare.common.sum(values, weights), 10.0)
    assert np.allclose(imod.prepare.common.minimum(values, weights), 1.0)
    assert np.allclose(imod.prepare.common.maximum(values, weights), 4.0)
    assert np.allclose(imod.prepare.common.median(values, weights), 2.5)
    assert np.allclose(imod.prepare.common.conductance(values, weights), 300.0)
    assert np.allclose(imod.prepare.common.max_overlap(values, weights), 4.0)

    # Check if no issues arise with all nan
    values[:] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        assert np.isnan(imod.prepare.common.sum(values, weights))
        assert np.isnan(imod.prepare.common.minimum(values, weights))
        assert np.isnan(imod.prepare.common.maximum(values, weights))
        assert np.isnan(imod.prepare.common.median(values, weights))
        assert np.isnan(imod.prepare.common.conductance(values, weights))
        assert np.isnan(imod.prepare.common.max_overlap(values, weights))


def test_methods_zeros():
    values = np.zeros(5)
    weights = np.arange(0.0, 50.0, 10.0)
    assert np.allclose(imod.prepare.common.mean(values, weights), 0.0)


def test_overlap():
    assert imod.prepare.common._overlap((0.0, 1.0), (0.0, 2.0)) == 1.0
    assert imod.prepare.common._overlap((-1.0, 1.0), (0.0, 2.0)) == 1.0
    assert imod.prepare.common._overlap((-1.0, 3.0), (0.0, 2.0)) == 2.0
    assert imod.prepare.common._overlap((-1.0, 3.0), (-2.0, 2.0)) == 3.0


def test_reshape():
    src = np.zeros((3, 5))
    dst = np.zeros((3, 2))
    iter_src, iter_dst = imod.prepare.common._reshape(src, dst, ndim_regrid=1)
    assert iter_src.shape == (3, 5)
    assert iter_dst.shape == (3, 2)

    src = np.zeros((2, 4, 3, 5))
    dst = np.zeros((2, 4, 3, 2))
    iter_src, iter_dst = imod.prepare.common._reshape(src, dst, ndim_regrid=1)
    assert iter_src.shape == (24, 5)
    assert iter_dst.shape == (24, 2)

    src = np.zeros((3, 5))
    dst = np.zeros((3, 2))
    iter_src, iter_dst = imod.prepare.common._reshape(src, dst, ndim_regrid=2)
    assert iter_src.shape == (1, 3, 5)
    assert iter_dst.shape == (1, 3, 2)

    src = np.zeros((2, 4, 3, 5))
    dst = np.zeros((2, 4, 3, 2))
    iter_src, iter_dst = imod.prepare.common._reshape(src, dst, ndim_regrid=3)
    assert iter_src.shape == (2, 4, 3, 5)
    assert iter_dst.shape == (2, 4, 3, 2)


def test_is_subset():
    # increasing
    a1 = np.array([0.0, 1.0, 2.0, 3.0])
    a2 = np.array([0.0, 1.0])
    assert imod.prepare.common._is_subset(a1, a2)
    a2 = np.array([0.0, 1.0, 3.0])
    assert not imod.prepare.common._is_subset(a1, a2)
    # decreasing
    a1 = np.array([0.0, 1.0, 2.0, 3.0])[::-1]
    a2 = np.array([0.0, 1.0])[::-1]
    assert imod.prepare.common._is_subset(a1, a2)
    a2 = np.array([0.0, 1.0, 3.0])[::-1]
    assert not imod.prepare.common._is_subset(a1, a2)
    # Not a contiguous subset
    a1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    a2 = np.array([0.0, 2.0, 4.0])
    assert not imod.prepare.common._is_subset(a1, a2)
    assert not imod.prepare.common._is_subset(a2, a1)


def test_selection_indices():
    # Left-inclusive
    # Vertices
    src_x = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    xmin, xmax = 0.0, 20.0
    i0, i1 = imod.prepare.common._selection_indices(src_x, xmin, xmax, 0)
    assert i0 == 0
    assert i1 == 1

    xmin, xmax = 0.0, 21.0
    i0, i1 = imod.prepare.common._selection_indices(src_x, xmin, xmax, 0)
    assert i0 == 0
    assert i1 == 2

    xmin, xmax = 0.0, 40.0
    i0, i1 = imod.prepare.common._selection_indices(src_x, xmin, xmax, 0)
    assert i0 == 0
    assert i1 == 2

    xmin, xmax = 0.0, 40.0
    i0, i1 = imod.prepare.common._selection_indices(src_x, xmin, xmax, 1)
    assert i0 == 0
    assert i1 == 3

    xmin, xmax = 20.0, 40.0
    i0, i1 = imod.prepare.common._selection_indices(src_x, xmin, xmax, 1)
    assert i0 == 0
    assert i1 == 3


def test_slice_src():
    matching_dims = ("x",)
    # dx of 100.0
    # midpoints
    src_x = np.array([50.0, 150.0, 250.0, 350.0, 450.0])
    # dx of 50.0
    like_x = np.array([75.0, 125.0, 175.0])
    src = xr.DataArray(np.ones(src_x.size), {"x": src_x}, ("x",))
    like = xr.DataArray(np.ones(like_x.size), {"x": like_x}, ("x",))

    actual = imod.prepare.common._slice_src(src, like, matching_dims, 0)
    expected = src.isel(x=slice(0, 2))
    assert actual.equals(expected)

    actual = imod.prepare.common._slice_src(src, like, matching_dims, 1)
    expected = src.isel(x=slice(0, 3))
    assert actual.equals(expected)

    like_x = np.array([125.0, 175.0, 225.0])
    like = xr.DataArray(np.ones(like_x.size), {"x": like_x}, ("x",))
    actual = imod.prepare.common._slice_src(src, like, matching_dims, 1)
    expected = src.isel(x=slice(0, 4))
    assert actual.equals(expected)


def test_define_single_dim_slices():
    # Simplest first, no chunk in dimension
    src_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    dst_x = np.array([0.0, 2.0, 4.0])
    chunksizes = (4,)
    dst_slices = imod.prepare.common._define_single_dim_slices(src_x, dst_x, chunksizes)
    assert dst_slices == [slice(None, None, None)]

    # Clean cuts
    chunksizes = (2, 2)
    dst_slices = imod.prepare.common._define_single_dim_slices(src_x, dst_x, chunksizes)
    assert dst_slices == [slice(0, 1, None), slice(1, 2, None)]

    # Mixed cut
    src_x = np.arange(13.0)
    dst_x = np.arange(0.0, 15.0, 2.5)
    chunksizes = (4, 4, 4)
    dst_slices = imod.prepare.common._define_single_dim_slices(src_x, dst_x, chunksizes)
    assert dst_slices == [slice(0, 2, None), slice(2, 4, None), slice(4, 5)]

    src_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    dst_x = np.array([0.0, 2.5, 5.0])
    chunksizes = (3, 2)
    dst_slices = imod.prepare.common._define_single_dim_slices(src_x, dst_x, chunksizes)
    assert dst_slices == [slice(0, 2, None)]

    # dst larger than src
    src_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    dst_x = np.array([-1.0, 2.5, 6.0])
    chunksizes = (3, 2)
    dst_slices = imod.prepare.common._define_single_dim_slices(src_x, dst_x, chunksizes)
    assert dst_slices == [slice(0, 2, None)]

    src_x = np.arange(13.0)
    dst_x = np.arange(0.0, 15.0, 2.5)
    chunksizes = (3, 3, 3, 3)
    dst_slices = imod.prepare.common._define_single_dim_slices(src_x, dst_x, chunksizes)
    assert dst_slices == [
        slice(0, 2, None),
        slice(2, 3, None),
        slice(3, 4, None),
        slice(4, 5, None),
    ]

    src_x = np.arange(0.0, 1010.0, 10.0)
    dst_x = np.arange(0.0, 1025.0, 25.0)
    chunksizes = (10,) * 10
    dst_slices = imod.prepare.common._define_single_dim_slices(src_x, dst_x, chunksizes)
    assert len(dst_slices) == 10

    src_x = np.arange(0.0, 400, 25.0)
    dst_x = np.arange(0.0, 390.0, 10.0)
    chunksizes = (3,) * 5
    dst_slices = imod.prepare.common._define_single_dim_slices(src_x, dst_x, chunksizes)
    assert len(dst_slices) == 5


def test_sel_chunks():
    src_x = np.arange(5.0) + 0.5
    dst_x = np.arange(0.0, 6.0, 2.0) + 1.0
    src = xr.DataArray(np.ones(5), {"x": src_x}, ("x",))
    src = src.chunk({"x": (2, 2, 1)})
    like = xr.DataArray(np.ones(3), {"x": dst_x}, ("x",))
    dst_slices, chunks_shape = imod.prepare.common._define_slices(src, like)
    assert len(dst_slices) == np.product(chunks_shape)

    # 2D
    src = xr.DataArray(np.ones((5, 5)), {"y": src_x, "x": src_x}, ("y", "x"))
    src = src.chunk({"x": (2, 2, 1), "y": (2, 2, 1)})
    like = xr.DataArray(np.ones((3, 3)), {"y": dst_x, "x": dst_x}, ("y", "x"))
    dst_slices, chunks_shape = imod.prepare.common._define_slices(src, like)
    assert len(dst_slices) == np.product(chunks_shape)
