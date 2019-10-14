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
        src_x, dst_x, True, False
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
        src_x, dst_x, True, False
    )
    assert max_len == 2
    assert np.allclose(dst_inds, np.array([0, 1]))
    assert np.allclose(src_inds, np.array([[0, 1], [1, 2]]))
    assert np.allclose(weights, np.array([[2.5, 2.5], [2.5, 2.5]]))

    # Mixed coords
    src_x = np.arange(-5.0, 6.0, 1.0)
    dst_x = np.arange(-5.0, 6.0, 2.5)
    max_len, (dst_inds, src_inds, weights) = imod.prepare.common._weights_1d(
        src_x, dst_x, True, False
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
        src_x, dst_x, True, True
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
        src_x, dst_x, True, True
    )
    assert np.allclose(weights, np.array([[1.0]]))

    src_x = np.array([0.0, 3.0])
    dst_x = np.array([0.0, 1.5])
    max_len, (dst_inds, src_inds, weights) = imod.prepare.common._weights_1d(
        src_x, dst_x, True, True
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


def test_is_increasing():
    src_x = np.arange(5.0)
    dst_x = np.arange(5.0)
    is_increasing = imod.prepare.common._is_increasing(src_x, dst_x)
    assert is_increasing

    src_x = np.arange(5.0, 0.0, -1.0)
    dst_x = np.arange(5.0, 0.0, -1.0)
    is_increasing = imod.prepare.common._is_increasing(src_x, dst_x)
    assert not is_increasing

    src_x = np.arange(5.0, 0.0, -1.0)
    dst_x = np.arange(5.0)
    with pytest.raises(ValueError):
        is_increasing = imod.prepare.common._is_increasing(src_x, dst_x)
