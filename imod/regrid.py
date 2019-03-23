import numba
import numpy as np
import xarray as xr


@numba.njit
def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


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


@numba.njit
def _weights_1d(src_x, dst_x):
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
    # i is index of dst
    # j is index of src
    for i, j in _starts(src_x, dst_x):
        dst_x0 = dst_x[i]
        dst_x1 = dst_x[i + 1]

        _inds = []
        _weights = []
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
                j += 1
        if has_value:
            dst_inds.append(i)
            src_inds.append(_inds)
            weights.append(_weights)
            # Save max number of source cells
            # So we know how much to pre-allocate later on
            inds_len = len(_inds)
            if inds_len > max_len:
                max_len = inds_len

    return max_len, (dst_inds, src_inds, weights)


@numba.njit
def _regrid_1d(src, dst, values, weights, inds_weights, method):
    """
    numba compiled function to regrid in three dimensions

    Parameters
    ----------
    src : np.array
    dst : np.array
    src_coords : tuple of np.arrays of edges
    dst_coords : tuple of np.arrays of edges
    method : numba.njit'ed function
    """
    kk, blocks_ix, blocks_weights_x = inds_weights[0]

    # i, j, k are indices of dst array
    # block_i contains indices of src array
    # block_w contains weights of src array
    for k, block_ix, block_wx in zip(kk, blocks_ix, blocks_weights_x):
        # Add the values and weights per cell in multi-dim block
        count = 0
        for ix, wx in zip(block_ix, block_wx):
            values[count] = src[ix]
            weights[count] = wx
            count += 1

        # aggregate
        dst[k] = method(values[:count], weights[:count])

        # reset storage
        values[:count] = 0
        weights[:count] = 0

    return dst


@numba.njit
def _regrid_2d(src, dst, values, weights, inds_weights, method):
    """
    numba compiled function to regrid in three dimensions

    Parameters
    ----------
    src : np.array
    dst : np.array
    src_coords : tuple of np.arrays of edges
    dst_coords : tuple of np.arrays of edges
    method : numba.njit'ed function
    """
    inds_weights_y, inds_weights_x = inds_weights
    jj, blocks_iy, blocks_weights_y = inds_weights_y
    kk, blocks_ix, blocks_weights_x = inds_weights_x

    # i, j, k are indices of dst array
    # block_i contains indices of src array
    # block_w contains weights of src array
    for j, block_iy, block_wy in zip(jj, blocks_iy, blocks_weights_y):
        for k, block_ix, block_wx in zip(kk, blocks_ix, blocks_weights_x):
            # Add the values and weights per cell in multi-dim block
            count = 0
            for iy, wy in zip(block_iy, block_wy):
                for ix, wx in zip(block_ix, block_wx):
                    values[count] = src[iy, ix]
                    weights[count] = wy * wx
                    count += 1

            # aggregate
            dst[j, k] = method(values[:count], weights[:count])

            # reset storage
            values[:count] = 0
            weights[:count] = 0

    return dst


@numba.njit
def _regrid_3d(src, dst, values, weights, inds_weights, method):
    """
    numba compiled function to regrid in three dimensions

    Parameters
    ----------
    src : np.array
    dst : np.array
    src_coords : tuple of np.arrays of edges
    dst_coords : tuple of np.arrays of edges
    method : numba.njit'ed function
    """
    inds_weights_z, inds_weights_y, inds_weights_x = inds_weights
    ii, blocks_iz, blocks_weights_z = inds_weights_z
    jj, blocks_iy, blocks_weights_y = inds_weights_y
    kk, blocks_ix, blocks_weights_x = inds_weights_x

    # i, j, k are indices of dst array
    # block_i contains indices of src array
    # block_w contains weights of src array
    for i, block_iz, block_wz in zip(ii, blocks_iz, blocks_weights_z):
        for j, block_iy, block_wy in zip(jj, blocks_iy, blocks_weights_y):
            for k, block_ix, block_wx in zip(kk, blocks_ix, blocks_weights_x):
                # Add the values and weights per cell in multi-dim block
                count = 0
                for iz, wz in zip(block_iz, block_wz):
                    for iy, wy in zip(block_iy, block_wy):
                        for ix, wx in zip(block_ix, block_wx):
                            values[count] = src[iz, iy, ix]
                            weights[count] = wz * wy * wx
                            count += 1

                # aggregate
                dst[i, j, k] = method(values[:count], weights[:count])

                # reset storage
                values[:count] = 0
                weights[:count] = 0

    return dst


@numba.njit
def _iter_regrid(iter_src, iter_dst, alloc_len, inds_weights, regrid_function):
    n_iter = iter_src.shape[0]
    # Pre-allocate temporary storage arrays
    values = np.zeros(alloc_len)
    weights = np.zeros(alloc_len)
    for i in range(n_iter):
        iter_dst[i, ...] = regrid_function(
            iter_src[i, ...], iter_dst[i, ...], values, weights, inds_weights
        )
    return iter_dst


def _jit_regrid(jit_method, ndim_regrid):
    """
    Compile a specific aggregation function using the compiled external method
    Closure avoids numba overhead
    https://numba.pydata.org/numba-doc/dev/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function
    """

    @numba.njit
    def jit_regrid_1d(src, dst, values, weights, inds_weights):
        return _regrid_1d(src, dst, values, weights, inds_weights, method=jit_method)

    @numba.njit
    def jit_regrid_2d(src, dst, values, weights, inds_weights):
        return _regrid_2d(src, dst, values, weights, inds_weights, method=jit_method)

    @numba.njit
    def jit_regrid_3d(src, dst, values, weights, inds_weights):
        return _regrid_3d(src, dst, values, weights, inds_weights, method=jit_method)

    if ndim_regrid == 1:
        jit_regrid = jit_regrid_1d
    elif ndim_regrid == 2:
        jit_regrid = jit_regrid_2d
    elif ndim_regrid == 3:
        jit_regrid = jit_regrid_3d
    else:
        raise NotImplementedError("cannot regrid over more than three dimensions")

    return jit_regrid


def _make_regrid(method, ndim_regrid):
    """
    Closure avoids numba overhead
    https://numba.pydata.org/numba-doc/dev/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function
    """

    # First, compile external method
    jit_method = numba.njit(method)
    jit_regrid = _jit_regrid(jit_method, ndim_regrid)

    # Finally, compile the iterating regrid method with the specific aggregation function
    @numba.njit
    def iter_regrid(iter_src, iter_dst, alloc_len, inds_weights):
        return _iter_regrid(
            iter_src,
            iter_dst,
            alloc_len,
            inds_weights,
            regrid_function=jit_regrid,
        )

    return iter_regrid


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


def _nd_regrid(src, dst, src_coords, dst_coords, iter_regrid):
    """
    Regrids an ndarray up to maximum 3 dimensions.
    Dimensionality of regridding is determined by the the length of src_coords
    (== len(dst_coords)), which has to match with the provide iter_regrid
    function.

    Parameters
    ----------
    src : np.array
    dst : np.array
    src_coords : tuple of np.array
    dst_coords : tuple of np.array
    iter_regrid : function, numba compiled
    """
    assert len(src.shape) == len(dst.shape)
    assert len(src_coords) == len(dst_coords)
    ndim_regrid = len(src_coords)

    # Determine weights for every regrid dimension, and alloc_len,
    # the maximum number of src cells that may end up in a single dst cell
    inds_weights = []
    alloc_len = 1
    for src_x, dst_x in zip(src_coords, dst_coords):
        s, i_w = _weights_1d(src_x, dst_x)
        # Convert to tuples so numba doesn't crash
        i_w = tuple(tuple(elem) for elem in i_w)
        inds_weights.append(i_w)
        alloc_len *= s
    inds_weights = tuple(inds_weights)

    iter_src, iter_dst = _reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, inds_weights)

    return iter_dst.reshape(dst.shape)


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
    for dim in src.dims():
        try:
            if src[dim].identical(like[dim]):
                matching_dims.append(dim)
            else:
                regrid_dims.append(dim)
        except KeyError:
            add_dims.append(dim)

    ndim_regrid = len(regrid_dims)
    # Check number of dimension to regrid
    if ndim_regrid == 0:
        return src
    if ndim_regrid > 3:
        raise NotImplementedError("cannot regrid over more than three dimensions")

    return matching_dims, regrid_dims, add_dims


def _slice_src(src, like, matching_dims):
    """
    Make sure src matches dst in dims that do not have to be regridded
    """

    for dim in matching_dims:
        x0 = like[dim][0]  # start of slice
        x1 = like[dim][-1]  # end of slice
        src = src.sel(dim=slice(x0, x1))
    return src


def _dst_coords(src, like, dims_from_src, dims_from_like):
    """
    Gather destination coordinates
    """

    dst_da_coords = {}
    dst_shape = []
    for dim in dims_from_src:
        dst_da_coords[dim] = src[dim].values
        dst_shape.append(src[dim].size)
    for dim in dims_from_like:
        dst_da_coords[dim] = like[dim].values
        dst_shape.append(like[dim].size)

    return dst_da_coords, dst_shape


def regrid(source, like, method, fill_value=np.nan):
    """
    Regridding for axis aligned coordinates
    Does both upscaling and downscaling (downsampling and upsampling, respectively)
    Equidistant and non-equidistant
    Custom aggregation methods can be defined

    Can regrid along 3 dimensions at a time.

    Parameters
    ----------
    source : xr.DataArray
        The source DataArray to be regridded
    like : xr.DataArray
        Example DataArray that shows what the resampled result should look like
        in terms of coordinates. `source` is regridded along dimensions of `like`
        that have the same name, but have different values.
    method : str, function
        The regridding method, for example mean, max, min.
        if str, one of default methods provided.
        if function, the function must take as arguments exactly (values, weights)
    fill_value : float64, optional
        Default value is np.nan

    Returns
    -------
    regridded : xr.DataArray
        `source` regridded along dimensions present in `like` with different
        values.

    Examples
    --------
    """
    # Don't mutate source; src stands for source, dst for destination
    src = source.copy()

    # Find coordinates that already match, and those that have to be regridded,
    # and those that exist in source but not in like (left untouched)
    matching_dims, regrid_dims, add_dims = _match_dims(src, like)

    # Make sure src matches dst in dims that do not have to be regridded
    src = _slice_src(src, like, matching_dims)

    # Order dimensions in the right way:
    # dimensions that are regridded end up at the end for efficient iteration
    dst_dims = (*add_dims, *matching_dims, *regrid_dims)
    dims_from_src = (*add_dims, *matching_dims)
    dims_from_like = tuple(regrid_dims)

    # Gather destination coordinates
    dst_da_coords, dst_shape = _dst_coords(src, like, dims_from_src, dims_from_like)

    # TODO: Check dimensionality of coordinates
    # 2-d coordinates should raise a ValueError
    # TODO: Check possibility to make gridding lazy
    # iter_regrid provides an opportunity for this, but the chunks need to be
    # defined somewhat intelligently: for 1d regridding for example the iter
    # loop is "hot" enough that numba compilation makes sense

    # Allocate dst
    # TODO: allocate lazy --> dask.array.full
    dst = xr.DataArray(
        data=np.full(dst_shape, fill_value), coords=dst_da_coords, dims=dst_dims
    )
    dst_coords_regrid = [dst[dim].values for dim in regrid_dims]
    src_coords_regrid = [src[dim].values for dim in regrid_dims]
    # Transpose src so that dims to regrid are last
    src = src.tranpose(*dst_dims)

    # Create tailor made regridding function: take method and ndims into account
    # and call it
    ndim_regrid = len(regrid_dims)
    iter_regrid = _make_regrid(method, ndim_regrid)
    dst.values = _nd_regrid(
        src.values,
        dst.values,
        src_coords_regrid,
        dst_coords_regrid,
        ndim_regrid,
        iter_regrid,
    )

    # Tranpose back to desired shape
    dst = dst.transpose(*source.dims)

    return dst


def mean(values, weights):
    vsum = 0.0
    wsum = 0.0
    for i in range(values.size):
        v = values[i]
        w = weights[i]
        vsum += w * v
        wsum += w
    return vsum / wsum


def harmonic_mean(values, weights):
    v_agg = 0.0
    w_sum = 0.0
    for i in range(values.size):
        w = weights[i]
        if w > 0:
            w_sum += w
            v_agg += w / values[i]
    return w_sum / v_agg


def geometric_mean(values, weights):
    v_agg = 0.0
    w_sum = 0.0
    m = 0
    for i in range(values.size):
        w = weights[i]
        v = values[i]
        if w > 0:
            v_agg += w * np.log(abs(v))
            w_sum += w
            if v < 0:
                m += 1
    return (-1.0) ** m * np.exp((1.0 / w_sum) * v_agg)


def sum(values, weights):
    v_sum = 0.0
    for i in range(values.size):
        v_sum += values[i] * weights[i]
    return v_sum


def minimum(values, weights):
    return np.min(values)


def maximum(values, weights):
    return np.max(values)


def mode(values, weights):
    ind_mode = np.argmax(np.bincount(values))
    return values[ind_mode]


def median(values, weights):
    return np.percentile(values, 50)
