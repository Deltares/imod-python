import os

import numba
import numpy as np
import pytest
import xarray as xr

import imod


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


