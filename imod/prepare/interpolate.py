"""
This module provides linear interpolation methods on regular grids, up to three 
dimensions. These functions are imported by the regrid.py module, which
incorporates them in the Regridder class.

### Introduction
The interp_ functions do most of the work. One dimension is added per time
(1, 2, 3). The simplest way to implement linear interpolation is as follows:

First, for one dimension. Let node 0 be located at x0, and have a value of v0,
and let node 1 be located at x1, and have a value of v1. Any value between node
0 and 1 can then be calculated as:

v(x) = v0 + (x - x0) / (x1 - x0) * (v1 - v0)

(x - x0) / (x1 - x0) is effectively a weighting of the different cells, and is
called wx in the code below.

Of course, we can rewrite it as:

v(x) = v0 + (x - x0) * (v1 - v0) / (x1 - x0)

Which is equal to the equation of a straight line with slope a, and intersect b.

b = v0
a = (delta_v / delta_x) = (v1 - v0) / (x1 - x0)

Multilinear interpolation can be implemented as repeated one dimensional linear
interpolation. Take two dimensions, with four cells. The values are located as
such:

  v00 -- v01
   |      |
   |      |
  v10 -- v11

v00 is upper left, v01 is upper right, etc. Given a point (x, y) between the
drawn boundaries, wx as above, and:
wy = (y - y0) / (y1 - y0)

Then, to compute the value for (x, y), we interpolate in y twice, and in x once.
(Or vice versa.)
v0 = v00 + wy * (v01 - v00)  # interpolate in y
v1 = v10 + wy * (v11 - v10)  # interpolate in y
v = v0 + wx * (v1 - v0)      # interpolate in x

### Nodata
All of this work perfectly, until we run into nodata values. The code below
is aimed at xarray.DataArrays, which use np.nan values as a sentinel for nodata.
NaN's pollute: 0 + nan = nan, 1 + 0 * nan = nan, etc.
In between cells that are not nodata, interpolation goes fine. However, at the
edges (say, to the left of v00 and v10), we get nan values. Note that the cell
of v00 does cover the area, but given a nan to left, it'll end up as nodata.

This is undesirable (the relative number of nodata cells increases).
We have to catch nan values before they pollute the computations. This is
straightforward in one dimension, but involves a bothersome number of
conditionals in 2d, and certainly in 3d! An easier implementation in this case
is not by doing repeated linear interpolations, but by using an accumulator,
and and an accumulating divisor. We simply add the contribution of every cell
to this accumulator, and skip it if it's nodata.

The result is the same (the algebra to prove this is left as an exercise to the
reader).

### Note quite there yet
Using the accumulator implementation, we end up creating more data, tiling over
nodata parts. Since we're always between two cells, and the implementation above
removes nodata values, non-nodata values will extend too far.

The easiest way of remedying this is by taking into account where the cell
boundaries of the original cell lie. If it's fully within a nodata cell, no
extrapolation should occur.

### Function responsibilities:
* linear_inds_weights: returns per single dimensions the weights (wx, wy), 
    the source indexes (column, row numbers), and whether the destination x
    lies fully within the cell of the matching source index. The source index
    is always the "starting" or "leftmost" cell. Interpolation occurs between
    this value and its neighbor to the right (index + 1).
* interp_ function implement functionality described above, taking the weights,
    indices, and the boolean within to compute interpolated values.
* iter_interp: an example best serves to explain: let there be a DataArray with
    dimensions time, layer, y, x. We wish to interpolate in two dimensions,
    over y and x. Time and layer are "stacked" onto a single dimension, so that
    a single loop in iter_interp suffices to interpolate over y and x, rather
    than a nested loop (in which case the number of for loops depends on the on
    the number of dimensions).
* jit_interp: selects the right interpolation method for the number of
    dimensions.
* make_interp: provides the closure to avoid numba overhead. Basically, we
    "inject" code before compilation, so we don't have to pass functions as an
    argument at runtime. (Numba probably inlines the function in this case.)
* nd_interp: collects the weights, reshapes the array so iter_interp will take
    it.
"""
import numba
import numpy as np

from imod.prepare import common


def _linear_inds_weights_1d(src_x, dst_x):
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
    # Start inclusive (just like e.g. GDAL)
    within = (mid_dst_x >= src_x[i]) & (mid_dst_x < src_x[i + 1])
    return i, norm_weights, within


@numba.njit(cache=True)
def _interp_1d(src, dst, *inds_weights):
    """
    Parameters
    ----------
    src : np.array
    dst : np.array
    """
    # Unpack the variadic arguments
    kk, weights_x, within_x = inds_weights
    # k are indices of dst array
    for k, (ix, wx, in_x) in enumerate(zip(kk, weights_x, within_x)):
        if ix < 0:
            continue

        # Fetch the values from source array, left v0, right v1
        v0 = src[ix]
        v1 = src[ix + 1]
        # Check whether they are nodata
        v0_ok = np.isfinite(v0)
        v1_ok = np.isfinite(v1)

        # Initialize and add to accumulators
        accumulator = 0
        accumulator_divisor = 0
        if v0_ok:
            multiplier = 1 - wx
            accumulator += multiplier * v0
            accumulator_divisor += multiplier
        if v1_ok:
            multiplier = wx
            accumulator += multiplier * v1
            accumulator_divisor += multiplier

        # Check if the point to interpolate to falls fully within a nodata cell
        # if that's the case, don't use the value, but continue with the next iteration.
        # else: use the value, fill it into the destination array.
        if accumulator_divisor > 0:
            if in_x:
                if not v0_ok:
                    continue
            else:
                if not v1_ok:
                    continue
            v = accumulator / accumulator_divisor
            dst[k] = v

    return dst


@numba.njit(cache=True)
def _interp_2d(src, dst, *inds_weights):
    # Unpack the variadic arguments
    jj, weights_y, within_y, kk, weights_x, within_x = inds_weights
    # j, k are indices of dst array
    for j, (iy, wy, in_y) in enumerate(zip(jj, weights_y, within_y)):
        if iy < 0:
            continue

        for k, (ix, wx, in_x) in enumerate(zip(kk, weights_x, within_x)):
            if ix < 0:
                continue

            # Fetch the values from source array, upper left v00, lower right v11
            v00 = src[iy, ix]
            v01 = src[iy, ix + 1]
            v10 = src[iy + 1, ix]
            v11 = src[iy + 1, ix + 1]
            # Check whether they are nodata
            v00_ok = np.isfinite(v00)
            v01_ok = np.isfinite(v01)
            v10_ok = np.isfinite(v10)
            v11_ok = np.isfinite(v11)

            # Initialize and add to accumulators
            accumulator = 0
            accumulator_divisor = 0
            if v00_ok:
                multiplier = (1 - wx) * (1 - wy)
                accumulator += multiplier * v00
                accumulator_divisor += multiplier
            if v01_ok:
                multiplier = wx * (1 - wy)
                accumulator += multiplier * v01
                accumulator_divisor += multiplier
            if v10_ok:
                multiplier = (1 - wx) * wy
                accumulator += multiplier * v10
                accumulator_divisor += multiplier
            if v11_ok:
                multiplier = wx * wy
                accumulator += multiplier * v11
                accumulator_divisor += multiplier

            # Check if the point to interpolate to falls fully within a nodata cell
            # if that's the case, don't use the value, but continue with the next iteration.
            # else: use the value, fill it into the destination array.
            if accumulator_divisor > 0:
                if in_y:
                    if in_x:
                        if not v00_ok:
                            continue
                    else:
                        if not v01_ok:
                            continue
                else:
                    if in_x:
                        if not v10_ok:
                            continue
                    else:
                        if not v11_ok:
                            continue
                v = accumulator / accumulator_divisor
                dst[j, k] = v

    return dst


@numba.njit(cache=True)
def _interp_3d(src, dst, *inds_weights):
    # Unpack the variadic arguments
    (
        ii,
        weights_z,
        within_z,
        jj,
        weights_y,
        within_y,
        kk,
        weights_x,
        within_x,
    ) = inds_weights
    # i, j, k are indices of dst array
    for i, (iz, wz, in_z) in enumerate(zip(ii, weights_z, within_z)):
        if iz < 0:
            continue

        for j, (iy, wy, in_y) in enumerate(zip(jj, weights_y, within_y)):
            if iy < 0:
                continue

            for k, (ix, wx, in_x) in enumerate(zip(kk, weights_x, within_x)):
                if ix < 0:
                    continue

                # Fetch the values from source array, top upper left v000,
                # bottom lower right v11
                v000 = src[iz, iy, ix]
                v001 = src[iz, iy, ix + 1]
                v010 = src[iz, iy + 1, ix]
                v011 = src[iz, iy + 1, ix + 1]
                v100 = src[iz + 1, iy, ix]
                v101 = src[iz + 1, iy, ix + 1]
                v110 = src[iz + 1, iy + 1, ix]
                v111 = src[iz + 1, iy + 1, ix + 1]
                # Check whether they are nodata
                v000_ok = np.isfinite(v000)
                v001_ok = np.isfinite(v001)
                v010_ok = np.isfinite(v010)
                v011_ok = np.isfinite(v011)
                v100_ok = np.isfinite(v100)
                v101_ok = np.isfinite(v101)
                v110_ok = np.isfinite(v110)
                v111_ok = np.isfinite(v111)

                # Initialize and add to accumulators
                accumulator = 0
                accumulator_divisor = 0
                if v000_ok:
                    multiplier = (1 - wz) * (1 - wx) * (1 - wy)
                    accumulator += multiplier * v000
                    accumulator_divisor += multiplier
                if v001_ok:
                    multiplier = (1 - wz) * wx * (1 - wy)
                    accumulator += multiplier * v001
                    accumulator_divisor += multiplier
                if v010_ok:
                    multiplier = (1 - wz) * (1 - wx) * wy
                    accumulator += multiplier * v010
                    accumulator_divisor += multiplier
                if v011_ok:
                    multiplier = (1 - wz) * wx * wy
                    accumulator += multiplier * v011
                    accumulator_divisor += multiplier
                if v100_ok:
                    multiplier = wz * (1 - wx) * (1 - wy)
                    accumulator += multiplier * v100
                    accumulator_divisor += multiplier
                if v101_ok:
                    multiplier = wz * wx * (1 - wy)
                    accumulator += multiplier * v101
                    accumulator_divisor += multiplier
                if v110_ok:
                    multiplier = wz * (1 - wx) * wy
                    accumulator += multiplier * v110
                    accumulator_divisor += multiplier
                if v111_ok:
                    multiplier = wz * wx * wy
                    accumulator += multiplier * v111
                    accumulator_divisor += multiplier

                # Check if the point to interpolate to falls fully within a nodata cell
                # if that's the case, don't use the value, but continue with the next iteration.
                # else: use the value, fill it into the destination array.
                if accumulator_divisor > 0:
                    if in_z:
                        if in_y:
                            if in_x:
                                if not v000_ok:
                                    continue
                            else:
                                if not v001_ok:
                                    continue
                        else:
                            if in_x:
                                if not v010_ok:
                                    continue
                            else:
                                if not v011_ok:
                                    continue
                    else:
                        if in_y:
                            if in_x:
                                if not v100_ok:
                                    continue
                            else:
                                if not v101_ok:
                                    continue
                        else:
                            if in_x:
                                if not v110_ok:
                                    continue
                            else:
                                if not v111_ok:
                                    continue
                    v = accumulator / accumulator_divisor
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
        for elem in _linear_inds_weights_1d(src_x, dst_x):
            inds_weights.append(elem)

    iter_src, iter_dst = common._reshape(src, dst, ndim_regrid)
    iter_dst = iter_interp(iter_src, iter_dst, *inds_weights)

    return iter_dst.reshape(dst.shape)
