"""
Common methods used for interpolation, voxelization.

Includes methods for dealing with different coordinates and dimensions of the
xarray.DataArrays, as well as aggregation methods operating on weights and
values.
"""
import numba
import numpy as np
import xarray as xr


@numba.njit(cache=True)
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


@numba.njit(cache=True)
def _weights_1d(src_x, dst_x, is_increasing, use_relative_weights=False):
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

    # Reverse the coordinate direction locally if coordinate is not
    # monotonically increasing, so starts and overlap continue to work.
    # copy() to avoid side-effects
    if not is_increasing:
        src_x = src_x.copy() * -1.0
        dst_x = dst_x.copy() * -1.0

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


def _reshape(src, dst, ndim_regrid):
    """
    If ndim > ndim_regrid, the non regridding dimension are combined into
    a single dimension, so we can use a single loop, irrespective of the
    total number of dimensions.
    (The alternative is pre-writing N for-loops for every N dimension we
    intend to support.)
    If ndims == ndim_regrid, all dimensions will be used in regridding
    in that case no looping over other dimensions is required and we add
    a dummy dimension here so there's something to iterate over.
    """
    src_shape = src.shape
    dst_shape = dst.shape
    ndim = len(src_shape)

    if ndim == ndim_regrid:
        n_iter = 1
    else:
        n_iter = int(np.product(src_shape[:-ndim_regrid]))

    src_itershape = (n_iter, *src_shape[-ndim_regrid:])
    dst_itershape = (n_iter, *dst_shape[-ndim_regrid:])

    iter_src = np.reshape(src, src_itershape)
    iter_dst = np.reshape(dst, dst_itershape)

    return iter_src, iter_dst


def _is_increasing(src_x, dst_x):
    """
    Make sure coordinate values always increase so the _starts function above
    works properly.
    """
    src_dx0 = src_x[1] - src_x[0]
    dst_dx0 = dst_x[1] - dst_x[0]
    if (src_dx0 > 0.0) ^ (dst_dx0 > 0.0):
        raise ValueError("source and like coordinates not in the same direction")
    if src_dx0 < 0.0:
        return False
    else:
        return True


def _match_dims(src, like):
    """
    Parameters
    ----------
    source : xr.DataArray
        The source DataArray to be regridded
    like : xr.DataArray
        Example DataArray that shows what the resampled result should look like
        in terms of coordinates. `source` is regridded along dimensions of `like`
        that have the same name, but have different values.

    Returns
    -------
    matching_dims, regrid_dims, add_dims : tuple of lists
        matching_dims: dimensions along which the coordinates match exactly
        regrid_dims: dimensions along which source will be regridded
        add_dims: dimensions that are not present in like

    """
    # TODO: deal with different extent?
    # Do another check if not identical
    # Check if subset or superset?
    matching_dims = []
    regrid_dims = []
    add_dims = []
    for dim in src.dims:
        try:
            if src[dim].identical(like[dim]):
                matching_dims.append(dim)
            else:
                regrid_dims.append(dim)
        except KeyError:
            add_dims.append(dim)

    ndim_regrid = len(regrid_dims)
    # Check number of dimension to regrid
    if ndim_regrid > 3:
        raise NotImplementedError("cannot regrid over more than three dimensions")

    return matching_dims, regrid_dims, add_dims


def _slice_src(src, like, matching_dims):
    """
    Make sure src matches dst in dims that do not have to be regridded
    """

    slices = {}
    for dim in matching_dims:
        x0 = like[dim][0]  # start of slice
        x1 = like[dim][-1]  # end of slice
        slices[dim] = slice(x0, x1)
    return src.sel(slices).compute()


def _dst_coords(src, like, dims_from_src, dims_from_like):
    """
    Gather destination coordinates
    """

    dst_da_coords = {}
    dst_shape = []
    # TODO: do some more checking, more robust handling
    like_coords = dict(like.coords)
    for dim in dims_from_src:
        try:
            like_coords.pop(dim)
        except KeyError:
            pass
        dst_da_coords[dim] = src[dim].values
        dst_shape.append(src[dim].size)
    for dim in dims_from_like:
        try:
            like_coords.pop(dim)
        except KeyError:
            pass
        dst_da_coords[dim] = like[dim].values
        dst_shape.append(like[dim].size)

    dst_da_coords.update(like_coords)
    return dst_da_coords, dst_shape


def _check_monotonic(dxs, dim):
    # use xor to check if one or the other
    if not ((dxs > 0.0).all() ^ (dxs < 0.0).all()):
        raise ValueError(f"{dim} is not only increasing or only decreasing")


def _coord(da, dim):
    delta_dim = "d" + dim  # e.g. dx, dy, dz, etc.

    if delta_dim in da.coords:  # equidistant or non-equidistant
        dx = da[delta_dim].values
        if dx.shape == () or dx.shape == (1,):  # scalar -> equidistant
            dxs = np.full(da[dim].size, dx)
        else:  # array -> non-equidistant
            dxs = dx
        _check_monotonic(dxs, dim)

    else:  # undefined -> equidistant
        dxs = np.diff(da[dim].values)
        dx = dxs[0]
        atolx = abs(1.0e-6 * dx)
        if not np.allclose(dxs, dx, atolx):
            raise ValueError(
                f"DataArray has to be equidistant along {dim}, or cellsizes"
                " must be provided as a coordinate."
            )
        dxs = np.full(da[dim].size, dx)

    # Check if the sign of dxs is correct for the coordinate values of x
    x = da[dim]
    dxs = np.abs(dxs)
    if x.size > 1:
        if x[1] < x[0]:
            dxs = -1.0 * dxs

    # Note: this works for both positive dx (increasing x) and negative dx
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


@numba.njit(cache=True)
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

    m = 0
    for i in range(values.size):
        w = weights[i] / normsum
        v = values[i]
        if np.isnan(v):
            continue
        if w > 0:
            v_agg += w * np.log(abs(v))
            w_sum += w
            if v < 0:
                m += 1

    if w_sum == 0:
        return np.nan
    else:
        return (-1.0) ** m * np.exp((1.0 / w_sum) * v_agg)


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


METHODS = {
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
