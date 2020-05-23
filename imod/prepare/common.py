"""
Common methods used for interpolation, voxelization.

Includes methods for dealing with different coordinates and dimensions of the
xarray.DataArrays, as well as aggregation methods operating on weights and
values.
"""
import cftime
import dask
import numba
import numpy as np
import xarray as xr

import imod


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


def _is_subset(a1, a2):
    if np.in1d(a2, a1).all():
        # This means all are present
        # now check if it's an actual subset
        # Generate number, and fetch only those present
        idx = np.arange(a1.size)[np.in1d(a1, a2)]
        if idx.size > 1:
            increment = np.diff(idx)
            # If the maximum increment is only 1, it's a subset
            if increment.max() == 1:
                return True
    return False


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
        if dim not in like.dims:
            add_dims.append(dim)
        elif src[dim].size == 0:  # zero overlap
            regrid_dims.append(dim)
        else:
            try:
                a1 = _coord(src, dim)
                a2 = _coord(like, dim)
                if np.array_equal(a1, a2) or _is_subset(a1, a2):
                    matching_dims.append(dim)
                else:
                    regrid_dims.append(dim)
            except TypeError:
                first_type = type(like[dim].values[0])
                if issubclass(first_type, (cftime.datetime, np.datetime64)):
                    raise RuntimeError(
                        "cannot regrid over datetime dimensions. "
                        "Use xarray.Dataset.resample() instead"
                    )

    ndim_regrid = len(regrid_dims)
    # Check number of dimension to regrid
    if ndim_regrid > 3:
        raise NotImplementedError("cannot regrid over more than three dimensions")

    return matching_dims, regrid_dims, add_dims


def _increasing_dims(da, dims):
    flip_dims = []
    for dim in dims:
        if not da.indexes[dim].is_monotonic_increasing:
            flip_dims.append(dim)
            da = da.isel({dim: slice(None, None, -1)})
    return da, flip_dims


def _selection_indices(src_x, xmin, xmax, extra_overlap):
    """Left-inclusive"""
    # Extra overlap is needed, for example with (multi)linear interpolation
    # We simply enlarge the slice at the start and at the end.
    i0 = max(0, np.searchsorted(src_x, xmin, side="right") - 1 - extra_overlap)
    i1 = np.searchsorted(src_x, xmax, side="left") + extra_overlap
    return i0, i1


def _slice_src(src, like, extra_overlap):
    """
    Make sure src matches dst in dims that do not have to be regridded
    """
    matching_dims, regrid_dims, _ = _match_dims(src, like)
    dims = matching_dims + regrid_dims

    slices = {}
    for dim in dims:
        # Generate vertices
        src_x = _coord(src, dim)
        _, xmin, xmax = imod.util.coord_reference(like[dim])
        i0, i1 = _selection_indices(src_x, xmin, xmax, extra_overlap)
        slices[dim] = slice(i0, i1)
    return src.isel(slices)


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


def _set_cellsizes(da, dims):
    for dim in dims:
        dx_string = f"d{dim}"
        if dx_string not in da:
            dx, _, _ = imod.util.coord_reference(da[dim])
            if isinstance(dx, (int, float)):
                dx = np.full(da[dim].size, dx)
            da = da.assign_coords({dx_string: (dim, dx)})
    return da


def _set_scalar_cellsizes(da):
    for dim in da.dims:
        dx_string = f"d{dim}"
        if dx_string in da:
            dx = da[dx_string]
            if np.allclose(dx, dx[0]):
                da = da.assign_coords({dx_string: dx[0]})
    return da


def _coord(da, dim):
    """
    Transform N xarray midpoints into N + 1 vertex edges
    """
    delta_dim = "d" + dim  # e.g. dx, dy, dz, etc.

    if delta_dim in da.coords:  # equidistant or non-equidistant
        dx = da[delta_dim].values
        if dx.shape == () or dx.shape == (1,):  # scalar -> equidistant
            dxs = np.full(da[dim].size, dx)
        else:  # array -> non-equidistant
            dxs = dx
        _check_monotonic(dxs, dim)

    else:  # undefined -> equidistant
        if da[dim].size == 1:
            raise ValueError(
                f"DataArray has size 1 along {dim}, so cellsize must be provided"
                " as a coordinate."
            )
        dxs = np.diff(da[dim].values)
        dx = dxs[0]
        atolx = abs(1.0e-6 * dx)
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


def _define_single_dim_slices(src_x, dst_x, chunksizes):
    n = len(chunksizes)
    assert n > 0
    if n == 1:
        return [slice(None, None)]

    chunk_indices = np.full(n + 1, 0)
    chunk_indices[1:] = np.cumsum(chunksizes)
    # Find locations to cut.
    src_chunk_x = src_x[chunk_indices]
    if dst_x[0] < src_chunk_x[0]:
        src_chunk_x[0] = dst_x[0]
    if dst_x[-1] > src_chunk_x[-1]:
        src_chunk_x[-1] = dst_x[-1]
    # Destinations should NOT have any overlap
    # Sources may have overlap
    # We find the most suitable places to cut.
    dst_i = np.searchsorted(dst_x, src_chunk_x, "left")
    dst_i[dst_i > dst_x.size - 1] = dst_x.size - 1

    # Create slices, but only if start and end are different
    # (otherwise, the slice would be empty)
    dst_slices = [slice(s, e) for s, e in zip(dst_i[:-1], dst_i[1:]) if s != e]
    return dst_slices


def _define_slices(src, like):
    """
    Defines the slices for every dimension, based on the chunks that are
    present within src.

    First, we get a single list of chunks per dimension.
    Next, these are expanded into an N-dimensional array, equal to the number
    of dimensions that have chunks.
    Finally, these arrays are ravelled, and stacked for easier iteration.
    """
    dst_dim_slices = []
    dst_chunks_shape = []
    for dim, chunksizes in zip(src.dims, src.chunks):
        if dim in like.dims:
            dst_slices = _define_single_dim_slices(
                _coord(src, dim), _coord(like, dim), chunksizes
            )
            dst_dim_slices.append(dst_slices)
            dst_chunks_shape.append(len(dst_slices))

    dst_expanded_slices = np.stack(
        [a.ravel() for a in np.meshgrid(*dst_dim_slices, indexing="ij")], axis=-1
    )
    return dst_expanded_slices, dst_chunks_shape


def _sel_chunks(da, expanded_slices):
    """
    Using the slices created with the functions above, use xarray's index
    selection methods to create a list of "like" DataArrays which are used
    to inform the regridding. During the regrid() call of the 
    imod.prepare.Regridder object, data from the input array is selected,
    ideally one chunk at time, or 2 ** ndim_chunks if there is overlap
    required due to cellsize differences.
    """
    das = []
    for dim_slices in expanded_slices:
        slice_dict = {}
        for dim, dim_slice in zip(da.dims, dim_slices):
            slice_dict[dim] = dim_slice
        das.append(da.isel(**slice_dict))
    return das


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
