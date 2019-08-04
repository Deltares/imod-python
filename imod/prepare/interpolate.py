import numba
import numpy as np

from imod.prepare.common import _is_increasing, _reshape


def _linear_inds_weights_1d(src_x, dst_x, is_increasing):
    """
    Returns indices and weights for linear interpolation along a single dimension.
    A sentinel value of -1 is added for dst cells that are fully out of bounds.

    Parameters
    ----------
    src_x : np.array
        vertex coordinates of source
    dst_x: np.array
        vertex coordinates of destination
    """
    if not is_increasing:
        src_x = src_x.copy() * -1.0
        dst_x = dst_x.copy() * -1.0
    xmin = src_x.min()
    xmax = src_x.max()

    # Compute midpoints for linear interpolation
    src_dx = np.diff(src_x)
    mid_src_x = src_x[:-1] + 0.5 * src_dx
    dst_dx = np.diff(dst_x)
    mid_dst_x = dst_x[:-1] + 0.5 * dst_dx

    # From np.searchsorted docstring:
    # Find the indices into a sorted array a such that, if the corresponding
    # elements in v were inserted before the indices, the order of a would
    # be preserved.
    i = np.searchsorted(mid_src_x, mid_dst_x) - 1
    # Out of bounds indices
    i[i < 0] = 0
    i[i > mid_src_x.size - 2] = mid_src_x.size - 2

    # -------------------------------------------------------------------------
    # Visual example: interpolate from src with 2 cells to dst 3 cells
    # The period . marks the midpoint of the cell
    # The pipe | marks the cell edge
    #
    #    |_____._____|_____._____|
    #    src_x0      src_x1
    #
    #    |___.___|___.___|___.___|
    #        x0      x1      x2
    #
    # Then normalized weight for cell x1:
    # weight = (x1 - src_x0) / (src_x1 - src_x0)
    # -------------------------------------------------------------------------

    norm_weights = (mid_dst_x - mid_src_x[i]) / (mid_src_x[i + 1] - mid_src_x[i])
    # deal with out of bounds locations
    # we place a sentinel value of -1 here
    i[mid_dst_x < xmin] = -1
    i[mid_dst_x > xmax] = -1
    # In case it's just inside of bounds, use only the value at the boundary
    norm_weights[norm_weights < 0.0] = 0.0
    norm_weights[norm_weights > 1.0] = 1.0
    # The following array is used only to deal with nodata values at the edges
    # Recall that src_x are the cell edges
    # Exclude the edges
    within = (mid_dst_x >= src_x[i]) & (mid_dst_x <= src_x[i + 1])
    start_edge = mid_dst_x == src_x[i]
    end_edge = mid_dst_x == src_x[i + 1]
    within = within.astype(np.int)
    within[start_edge] = -1
    within[end_edge] = -2
    return i, norm_weights, within


@numba.njit(cache=True)
def _interp_1d(src, dst, *inds_weights):
    """
    Parameters
    ----------
    src : np.array
    dst : np.array
    """
    kk, weights_x, within_x = inds_weights
    # i, j, k are indices of dst array
    for k, (ix, wx, in_x) in enumerate(zip(kk, weights_x, within_x)):
        if ix < 0:
            continue
        v0 = src[ix]
        v1 = src[ix + 1]
        v, ok = _catch_nan(v0, v1, wx, in_x)
        if ok:  # else: this value is skipped
            dst[k] = v
    return dst


@numba.njit(cache=True)
def _interp_2d(src, dst, *inds_weights):
    jj, weights_y, within_y, kk, weights_x, within_x = inds_weights

    for j, (iy, wy, in_y) in enumerate(zip(jj, weights_y, within_y)):
        if iy < 0:
            continue

        for k, (ix, wx, in_x) in enumerate(zip(kk, weights_x, within_x)):
            if ix < 0:
                continue

            v00 = src[iy, ix]
            v01 = src[iy, ix + 1]
            v10 = src[iy + 1, ix]
            v11 = src[iy + 1, ix + 1]

            accumulator = 0
            accumulator_divisor = 0

            if ~np.isnan(v00):
                multiplier = (1 - wx) * (1 - wy)
                accumulator += multiplier * v00
                accumulator_divisor += multiplier
            if ~np.isnan(v01):
                multiplier = wx * (1 - wy)
                accumulator += multiplier * v01
                accumulator_divisor += multiplier
            if ~np.isnan(v10):
                multiplier = (1 - wx) * wy
                accumulator += multiplier * v10
                accumulator_divisor += multiplier
            if ~np.isnan(v11):
                multiplier = wx * wy
                accumulator += multiplier * v11
                accumulator_divisor += multiplier
            
            if accumulator_divisor > 0.0:
                v = accumulator / accumulator_divisor
                dst[j, k] = v

    return dst


@numba.njit(cache=True)
def _interp_3d(src, dst, *inds_weights):
    ii, weights_z, within_z, jj, weights_y, within_y, kk, weights_x, within_x = inds_weights
    for i, (iz, wz) in enumerate(zip(ii, weights_z)):
        if iz < 0:
            continue

        for j, (iy, wy) in enumerate(zip(jj, weights_y)):
            if iy < 0:
                continue

            for k, (ix, wx) in enumerate(zip(kk, weights_x)):
                if ix < 0:
                    continue

                v000 = src[iz, iy, ix]
                v001 = src[iz, iy, ix + 1]
                v010 = src[iz, iy + 1, ix]
                v011 = src[iz, iy + 1, ix + 1]
                v100 = src[iz + 1, iy, ix]
                v101 = src[iz + 1, iy, ix + 1]
                v110 = src[iz + 1, iy + 1, ix]
                v111 = src[iz + 1, iy + 1, ix + 1]

                # First interpolate over z
                v00 = v000 + wz * (v100 - v000)
                v01 = v000 + wz * (v101 - v001)
                v10 = v000 + wz * (v110 - v010)
                v11 = v000 + wz * (v111 - v011)
                # Second interpolate over y
                v0 = v00 + wy * (v10 - v00)
                v1 = v01 + wy * (v11 - v01)
                # Third interpolate over x
                v = v0 + wx * (v1 - v0)

                dst[i, j, k] = v
    return dst


@numba.njit
def _iter_interp(iter_src, iter_dst, interp_function, *inds_weights):
    n_iter = iter_src.shape[0]
    for i in range(n_iter):
        iter_dst[i, ...] = interp_function(
            iter_src[i, ...], iter_dst[i, ...], *inds_weights
        )
    return iter_dst


def _jit_interp(ndim_interp):
    @numba.njit
    def jit_interp_1d(src, dst, *inds_weights):
        return _interp_1d(src, dst, *inds_weights)

    @numba.njit
    def jit_interp_2d(src, dst, *inds_weights):
        return _interp_2d(src, dst, *inds_weights)

    @numba.njit
    def jit_interp_3d(src, dst, *inds_weights):
        return _interp_3d(src, dst, *inds_weights)

    if ndim_interp == 1:
        jit_interp = jit_interp_1d
    elif ndim_interp == 2:
        jit_interp = jit_interp_2d
    elif ndim_interp == 3:
        jit_interp = jit_interp_3d
    else:
        raise NotImplementedError("cannot regrid over more than three dimensions")

    return jit_interp


def _make_interp(ndim_regrid):
    jit_interp = _jit_interp(ndim_regrid)

    @numba.njit
    def iter_interp(iter_src, iter_dst, *inds_weights):
        return _iter_interp(iter_src, iter_dst, jit_interp, *inds_weights)

    return iter_interp


def _nd_interp(src, dst, src_coords, dst_coords, iter_interp):
    assert len(src.shape) == len(dst.shape)
    assert len(src_coords) == len(dst_coords)
    ndim_regrid = len(src_coords)

    # Determine weights for every regrid dimension, and alloc_len,
    # the maximum number of src cells that may end up in a single dst cell
    inds_weights = []
    for src_x, dst_x in zip(src_coords, dst_coords):
        is_increasing = _is_increasing(src_x, dst_x)
        iw = _linear_inds_weights_1d(src_x, dst_x, is_increasing)
        for elem in iw:
            inds_weights.append(elem)

    iter_src, iter_dst = _reshape(src, dst, ndim_regrid)
    iter_dst = iter_interp(iter_src, iter_dst, *inds_weights)

    return iter_dst.reshape(dst.shape)
