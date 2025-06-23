"""
Common methods used for interpolation, voxelization.

Includes methods for dealing with different coordinates and dimensions of the
xarray.DataArrays, as well as aggregation methods operating on weights and
values.
"""

from typing import Any

import numba
import numpy as np


@numba.njit
def _starts(src_x, dst_x):
    """
    Calculate regridding weights for a single dimension

    Parameters
    ----------
    src_x : np.array
        vertex coordinates of source
    dst_x: np.array
        vertex coordinates of destination
    """
    i = 0
    j = 0
    while i < dst_x.size - 1:
        x = dst_x[i]
        while j < src_x.size:
            if src_x[j] > x:
                out = max(j - 1, 0)
                yield (i, out)
                break
            else:
                j += 1
        i += 1


def _weights_1d(src_x, dst_x, use_relative_weights=False):
    """
    Calculate regridding weights and indices for a single dimension

    Parameters
    ----------
    src_x : np.array
        vertex coordinates of source
    dst_x: np.array
        vertex coordinates of destination

    Returns
    -------
    max_len : int
        maximum number of source cells to a single destination cell for this
        dimension
    dst_inds : list of int
        destination cell index
    src_inds: list of list of int
        source cell index, per destination index
    weights : list of list of float
        weight of source cell, per destination index
    """
    max_len = 0
    dst_inds = []
    src_inds = []
    weights = []
    rel_weights = []

    # i is index of dst
    # j is index of src
    for i, j in _starts(src_x, dst_x):
        dst_x0 = dst_x[i]
        dst_x1 = dst_x[i + 1]

        _inds = []
        _weights = []
        _rel_weights = []
        has_value = False
        while j < src_x.size - 1:
            src_x0 = src_x[j]
            src_x1 = src_x[j + 1]
            overlap = _overlap((dst_x0, dst_x1), (src_x0, src_x1))
            # No longer any overlap, continue to next dst cell
            if overlap == 0:
                break
            else:
                has_value = True
                _inds.append(j)
                _weights.append(overlap)
                relative_overlap = overlap / (src_x1 - src_x0)
                _rel_weights.append(relative_overlap)
                j += 1
        if has_value:
            dst_inds.append(i)
            src_inds.append(_inds)
            weights.append(_weights)
            rel_weights.append(_rel_weights)
            # Save max number of source cells
            # So we know how much to pre-allocate later on
            inds_len = len(_inds)
            if inds_len > max_len:
                max_len = inds_len

    # Convert all output to numpy arrays
    # numba does NOT like arrays or lists in tuples
    # Compilation time goes through the roof
    nrow = len(dst_inds)
    ncol = max_len
    np_dst_inds = np.array(dst_inds)

    np_src_inds = np.full((nrow, ncol), -1)
    for i in range(nrow):
        for j, ind in enumerate(src_inds[i]):
            np_src_inds[i, j] = ind

    np_weights = np.full((nrow, ncol), 0.0)
    if use_relative_weights:
        weights = rel_weights
    for i in range(nrow):
        for j, ind in enumerate(weights[i]):
            np_weights[i, j] = ind

    return max_len, (np_dst_inds, np_src_inds, np_weights)


def _is_subset(a1, a2):
    if np.isin(a2, a1).all():
        # This means all are present
        # now check if it's an actual subset
        # Generate number, and fetch only those present
        idx = np.arange(a1.size)[np.isin(a1, a2)]
        if idx.size > 1:
            increment = np.diff(idx)
            # If the maximum increment is only 1, it's a subset
            if increment.max() == 1:
                return True
    return False


def _increasing_dims(da, dims):
    flip_dims = []
    for dim in dims:
        if not da.indexes[dim].is_monotonic_increasing:
            flip_dims.append(dim)
            da = da.isel({dim: slice(None, None, -1)})
    return da, flip_dims


def _check_monotonic(dxs, dim):
    # use xor to check if one or the other
    if not ((dxs > 0.0).all() ^ (dxs < 0.0).all()):
        raise ValueError(f"{dim} is not only increasing or only decreasing")


def _coord(da, dim):
    """
    Transform N xarray midpoints into N + 1 vertex edges
    """
    delta_dim = "d" + dim  # e.g. dx, dy, dz, etc.

    # If empty array, return empty
    if da[dim].size == 0:
        return np.array(())

    if delta_dim in da.coords:  # equidistant or non-equidistant
        dx = da[delta_dim].values
        if dx.shape == () or dx.shape == (1,):  # scalar -> equidistant
            dxs = np.full(da[dim].size, dx)
        else:  # array -> non-equidistant
            dxs = dx
        _check_monotonic(dxs, dim)

    else:  # not defined -> equidistant
        if da[dim].size == 1:
            raise ValueError(
                f"DataArray has size 1 along {dim}, so cellsize must be provided"
                " as a coordinate."
            )
        dxs = np.diff(da[dim].values)
        dx = dxs[0]
        atolx = abs(1.0e-4 * dx)
        if not np.allclose(dxs, dx, atolx):
            raise ValueError(
                f"DataArray has to be equidistant along {dim}, or cellsizes"
                " must be provided as a coordinate."
            )
        dxs = np.full(da[dim].size, dx)

    dxs = np.abs(dxs)
    x = da[dim].values
    if not da.indexes[dim].is_monotonic_increasing:
        x = x[::-1]
        dxs = dxs[::-1]

    # This assumes the coordinate to be monotonic increasing
    x0 = x[0] - 0.5 * dxs[0]
    x = np.full(dxs.size + 1, x0)
    x[1:] += np.cumsum(dxs)
    return x


def _get_method(method, methods):
    if isinstance(method, str):
        try:
            _method = methods[method]
        except KeyError as e:
            raise ValueError(
                "Invalid regridding method. Available methods are: {}".format(
                    methods.keys()
                )
            ) from e
    elif callable(method):
        _method = method
    else:
        raise TypeError("method must be a string or rasterio.enums.Resampling")
    return _method


@numba.njit
def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def mean(values, weights):
    vsum = 0.0
    wsum = 0.0
    for i in range(values.size):
        v = values[i]
        w = weights[i]
        if np.isnan(v):
            continue
        vsum += w * v
        wsum += w
    if wsum == 0:
        return np.nan
    else:
        return vsum / wsum


def harmonic_mean(values, weights):
    v_agg = 0.0
    w_sum = 0.0
    for i in range(values.size):
        v = values[i]
        w = weights[i]
        if np.isnan(v) or v == 0:
            continue
        if w > 0:
            w_sum += w
            v_agg += w / v
    if v_agg == 0 or w_sum == 0:
        return np.nan
    else:
        return w_sum / v_agg


def geometric_mean(values, weights):
    v_agg = 0.0
    w_sum = 0.0

    # Compute sum to ormalize weights to avoid tiny or huge values in exp
    normsum = 0.0
    for i in range(values.size):
        normsum += weights[i]
    # Early return if no values
    if normsum == 0:
        return np.nan

    for i in range(values.size):
        w = weights[i] / normsum
        v = values[i]
        # Skip if v == 0, v is NaN or w == 0 (no contribution)
        if v > 0 and w > 0:
            v_agg += w * np.log(abs(v))
            w_sum += w
        # Do not reduce over negative values: would require complex numbers.
        elif v < 0:
            return np.nan

    if w_sum == 0:
        return np.nan
    else:
        return np.exp((1.0 / w_sum) * v_agg)


def sum(values, weights):
    v_sum = 0.0
    w_sum = 0.0
    for i in range(values.size):
        v = values[i]
        w = weights[i]
        if np.isnan(v):
            continue
        v_sum += v
        w_sum += w
    if w_sum == 0:
        return np.nan
    else:
        return v_sum


def minimum(values, weights):
    return np.nanmin(values)


def maximum(values, weights):
    return np.nanmax(values)


def mode(values, weights):
    # Area weighted mode
    # Reuse weights to do counting: no allocations
    # The alternative is defining a separate frequency array in which to add
    # the weights. This implementation is less efficient in terms of looping.
    # With many unique values, it keeps having to loop through a big part of
    # the weights array... but it would do so with a separate frequency array
    # as well. There are somewhat more elements to traverse in this case.
    s = values.size
    w_sum = 0
    for i in range(s):
        v = values[i]
        w = weights[i]
        if np.isnan(v):
            continue
        w_sum += 1
        for j in range(i):  # Compare with previously found values
            if values[j] == v:  # matches previous value
                weights[j] += w  # increase previous weight
                break

    if w_sum == 0:  # It skipped everything: only nodata values
        return np.nan
    else:  # Find value with highest frequency
        w_max = 0
        for i in range(s):
            w = weights[i]
            if w > w_max:
                w_max = w
                v = values[i]
        return v


def median(values, weights):
    return np.nanpercentile(values, 50)


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


def max_overlap(values, weights):
    max_w = 0.0
    v = np.nan
    for i in range(values.size):
        w = weights[i]
        if w > max_w:
            max_w = w
            v = values[i]
    return v


METHODS: dict[str, Any] = {
    "nearest": "nearest",
    "multilinear": "multilinear",
    "mean": mean,
    "harmonic_mean": harmonic_mean,
    "geometric_mean": geometric_mean,
    "sum": sum,
    "minimum": minimum,
    "maximum": maximum,
    "mode": mode,
    "median": median,
    "conductance": conductance,
    "max_overlap": max_overlap,
}
