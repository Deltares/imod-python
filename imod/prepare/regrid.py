import warnings

import numba
import numpy as np
import xarray as xr


def _linear_inds_weights_1d(src_x, dst_x, xmin, xmax):
    """
    Returns indices and weights for linear interpolation along a single dimension.
    A sentinel value of -1 is added for dst cells that are fully out of bounds.

    Parameters
    ----------
    src_x : np.array
        Location of points on source grid. NOT vertex locations!
    dst_x : np.array
        Location of points on destination grid. NOT vertex locations!
    xmin : float
        Minimum coordinate value for src
    xmax : float
        Maximum coordinate value for src
    """
    # From np.searchsorted docstring:
    # Find the indices into a sorted array a such that, if the corresponding
    # elements in v were inserted before the indices, the order of a would
    # be preserved.
    i = np.searchsorted(src_x, dst_x) - 1
    # Out of bounds indices
    i[i < 0] = 0
    i[i > src_x.size - 2] = src_x.size - 2

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

    norm_weights = (dst_x - src_x[i]) / (src_x[i + 1] - src_x[i])
    # deal with out of bounds locations
    # we place a sentinel value of -1 here
    i[dst_x < xmin] = -1
    i[dst_x > xmax] = -1
    # In case it's just inside of bounds, use only the value at the boundary
    norm_weights[norm_weights < 0.0] = 0.0
    norm_weights[norm_weights > 1.0] = 1.0
    return i, norm_weights


@numba.njit
def _iter_interpolate(src, dst, inds, weights):
    """
    Parameters
    ----------
    src : np.array
        source array, reshaped to a 2d array. Interpolation occurs over the
        last dimension.
    dst : np.array
        destination array, reshaped to a 2d array.
    inds : np.array
        indexes, size equal to last dimension of dst
    weights : np.array
        normalized weight, size equal to last dimension of dst

    Returns
    -------
    dst : np.array
        destination, contains interpolated values over last dimension
    """
    n_iter, _ = src.shape
    for i in range(n_iter):
        for j, ind in enumerate(inds):
            if ind < 0:  # sentinel value: out of bounds
                continue
            v0 = src[i, ind]
            v1 = src[i, ind + 1]
            dst[i, j] = v0 + weights[j] * (v1 - v0)
            # See _linear_inds_weights_1d for explanation of weight
    return dst


def _nd_interp(src, inds, weights, fill_value=np.nan):
    # Make sure we don't mutate source
    temp = src.copy()
    dst = temp
    # Get all the necessary shape information
    orig_shape = src.shape
    ndim_regrid = len(inds)
    new_shape = orig_shape[:-ndim_regrid] + tuple([i.size for i in inds])
    # Temp shape for bookkeeping, where which axis ends up
    temp_shape = list(new_shape)

    for count, (i, w) in enumerate(zip(inds, weights)):
        if count > 0:
            # Move interpolated axis to the start
            # Expose next axis at the end to interpolate
            temp = np.moveaxis(temp, -1, 0)
            temp_shape.insert(0, temp_shape.pop(-1))

        # Allocate destination
        dst = np.full((*temp.shape[:-1], i.size), fill_value)
        # ndim_regrid for linear interpolation is always one
        temp, dst = _reshape(temp, dst, ndim_regrid=1)
        # Interpolate over a single dimension
        dst = _iter_interpolate(temp, dst, i, w)
        temp = dst.reshape(orig_shape[: count - 1] + new_shape[count - 1 :])

    # Restore to original shape
    # Doesn't allocate
    # TODO: maybe call .ascontiguousarray?
    dst = dst.reshape(temp_shape)
    for _ in range(count):
        dst = np.moveaxis(dst, 0, -1)

    return dst


@numba.njit(cache=True)
def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


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


@numba.njit(cache=True)
def _regrid_1d(src, dst, values, weights, method, *inds_weights):
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
    kk, blocks_ix, blocks_weights_x = inds_weights
    # i, j, k are indices of dst array
    # block_i contains indices of src array
    # block_w contains weights of src array
    for countk, k in enumerate(kk):
        block_ix = blocks_ix[countk]
        block_wx = blocks_weights_x[countk]
        # Add the values and weights per cell in multi-dim block
        count = 0
        for ix, wx in zip(block_ix, block_wx):
            if ix < 0:
                break
            values[count] = src[ix]
            weights[count] = wx
            count += 1

        # aggregate
        dst[k] = method(values[:count], weights[:count])

        # reset storage
        values[:count] = 0
        weights[:count] = 0

    return dst


@numba.njit(cache=True)
def _regrid_2d(src, dst, values, weights, method, *inds_weights):
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
    jj, blocks_iy, blocks_weights_y, kk, blocks_ix, blocks_weights_x = inds_weights

    # i, j, k are indices of dst array
    # block_i contains indices of src array
    # block_w contains weights of src array
    for countj, j in enumerate(jj):
        block_iy = blocks_iy[countj]
        block_wy = blocks_weights_y[countj]
        for countk, k in enumerate(kk):
            block_ix = blocks_ix[countk]
            block_wx = blocks_weights_x[countk]
            # Add the values and weights per cell in multi-dim block
            count = 0
            for iy, wy in zip(block_iy, block_wy):
                if iy < 0:
                    break
                for ix, wx in zip(block_ix, block_wx):
                    if ix < 0:
                        break
                    values[count] = src[iy, ix]
                    weights[count] = wy * wx
                    count += 1

            # aggregate
            dst[j, k] = method(values[:count], weights[:count])

            # reset storage
            values[:count] = 0.0
            weights[:count] = 0.0

    return dst


@numba.njit(cache=True)
def _regrid_3d(src, dst, values, weights, method, *inds_weights):
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
    ii, blocks_iz, blocks_weights_z, jj, blocks_iy, blocks_weights_y, kk, blocks_ix, blocks_weights_x = (
        inds_weights
    )

    # i, j, k are indices of dst array
    # block_i contains indices of src array
    # block_w contains weights of src array
    for counti, i in enumerate(ii):
        block_iz = blocks_iz[counti]
        block_wz = blocks_weights_z[counti]
        for countj, j in enumerate(jj):
            block_iy = blocks_iy[countj]
            block_wy = blocks_weights_y[countj]
            for countk, k in enumerate(kk):
                block_ix = blocks_ix[countk]
                block_wx = blocks_weights_x[countk]
                # Add the values and weights per cell in multi-dim block
                count = 0
                for iz, wz in zip(block_iz, block_wz):
                    if iz < 0:
                        break
                    for iy, wy in zip(block_iy, block_wy):
                        if iy < 0:
                            break
                        for ix, wx in zip(block_ix, block_wx):
                            if ix < 0:
                                break
                            values[count] = src[iz, iy, ix]
                            weights[count] = wz * wy * wx
                            count += 1

                # aggregate
                dst[i, j, k] = method(values[:count], weights[:count])

                # reset storage
                values[:count] = 0.0
                weights[:count] = 0.0

    return dst


@numba.njit
def _iter_regrid(iter_src, iter_dst, alloc_len, regrid_function, *inds_weights):
    n_iter = iter_src.shape[0]
    # Pre-allocate temporary storage arrays
    values = np.zeros(alloc_len)
    weights = np.zeros(alloc_len)
    for i in range(n_iter):
        iter_dst[i, ...] = regrid_function(
            iter_src[i, ...], iter_dst[i, ...], values, weights, *inds_weights
        )
    return iter_dst


def _jit_regrid(jit_method, ndim_regrid):
    """
    Compile a specific aggregation function using the compiled external method
    Closure avoids numba overhead
    https://numba.pydata.org/numba-doc/dev/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function
    """

    @numba.njit
    def jit_regrid_1d(src, dst, values, weights, *inds_weights):
        return _regrid_1d(src, dst, values, weights, jit_method, *inds_weights)

    @numba.njit
    def jit_regrid_2d(src, dst, values, weights, *inds_weights):
        return _regrid_2d(src, dst, values, weights, jit_method, *inds_weights)

    @numba.njit
    def jit_regrid_3d(src, dst, values, weights, *inds_weights):
        return _regrid_3d(src, dst, values, weights, jit_method, *inds_weights)

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
    jit_method = numba.njit(method, cache=True)
    jit_regrid = _jit_regrid(jit_method, ndim_regrid)

    # Finally, compile the iterating regrid method with the specific aggregation function
    @numba.njit
    def iter_regrid(iter_src, iter_dst, alloc_len, *inds_weights):
        return _iter_regrid(iter_src, iter_dst, alloc_len, jit_regrid, *inds_weights)

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


def _nd_regrid(src, dst, src_coords, dst_coords, iter_regrid, use_relative_weights):
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
        is_increasing = _is_increasing(src_x, dst_x)
        size, i_w = _weights_1d(src_x, dst_x, is_increasing, use_relative_weights)
        for elem in i_w:
            inds_weights.append(elem)
        alloc_len *= size

    iter_src, iter_dst = _reshape(src, dst, ndim_regrid)
    iter_dst = iter_regrid(iter_src, iter_dst, alloc_len, *inds_weights)

    return iter_dst.reshape(dst.shape)


def _match_dims(src, like):
    """
    Parameters
    ----------
    source : xr.DataArray
        The source DataArray to be regridded
    like : xr.DataArray
        Example DataArray that shows what the resampled result should look like
        in terms of coordinates. ``source`` is regridded along dimensions of ``like``


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
    if ndim_regrid == 0:
        return src
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

    else:  # undefined -> equidistant
        dxs = np.diff(da[dim].values)
        dx = dxs[0]
        atolx = abs(1.0e-6 * dx)
        if not np.allclose(dxs, dx, atolx):
            raise ValueError(
                f"DataArray has to be equidistant along {dim}, or d{dim} must"
                " be provided as a coordinate."
            )
        dxs = np.full(da[dim].size, dx)

    _check_monotonic(dxs, dim)
    x0 = da[dim][0] - 0.5 * dxs[0]
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


class Regridder(object):
    """
    Object to repeatedly regrid similar objects. Compiles once on first call,
    can then be repeatedly called without JIT compilation overhead.

    Attributes
    ----------
    method : str, function
        The method to use for regridding. Default available methods are:
        {"mean", "harmonic_mean", "geometric_mean", "sum", "minimum",
        "maximum", "mode", "median", "conductance"}
    ndim_regrid : int, optional
        The number of dimensions over which to regrid. If not provided,
        ``ndim_regrid`` will be inferred. It serves to prevent regridding over an
        unexpected number of dimensions; say you want to regrid over only two
        dimensions. Due to an input error in the coordinates of ``like``, three
        dimensions may be inferred in the first ``.regrid`` call. An error will
        be raised if ndim_regrid not match the number of inferred dimensions.
        Default value is None.
    use_relative_weights : bool, optional
        Whether to use relative weights in the regridding method or not.
        Relative weights are defined as: cell_overlap / source_cellsize, for
        every axis.

        This argument should only be used if you are providing your own
        ``method`` as a function, where the function requires relative, rather
        than absolute weights (the provided ``conductance`` method requires
        relative weights, for example). Default value is False.

    Examples
    --------
    Initialize the Regridder object:

    >>> mean_regridder = imod.prepare.Regridder(method="mean")

    Then call the ``regrid`` method to regrid.

    >>> result = mean_regridder(source, like)

    The regridder can be re-used if the number of regridding dimensions
    match, saving some time by not (re)compiling the regridding method.

    >>> second_result = mean_regrid(second_source, like)

    A one-liner is possible for single use:

    >>> result = imod.prepare.Regridder(method="mean").regrid(source, like)

    It's possible to provide your own methods to the ``Regridder``, provided that
    numba can compile them. They need to take the arguments ``values`` and
    ``weights``. Make sure they deal with nan values gracefully!

    >>> def p30(values, weights):
    >>>     return np.nanpercentile(values, 30)

    >>> p30_regridder = imod.prepare.Regridder(method=p30)
    >>> p30_result = p30_regridder.regrid(source, like)

    The Numba developers maintain a list of support Numpy features here:
    https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html

    In general, however, the provided methods should be adequate for your
    regridding needs.
    """

    def __init__(self, method, ndim_regrid=None, use_relative_weights=False):
        _method = _get_method(method, METHODS)
        self.method = _method
        self.ndim_regrid = ndim_regrid
        self._first_call = True
        if _method == conductance:
            use_relative_weights = True
        self.use_relative_weights = use_relative_weights

    def _make_regrid(self):
        iter_regrid = _make_regrid(self.method, self.ndim_regrid)

        def nd_regrid(src, dst, src_coords_regrid, dst_coords_regrid):
            return _nd_regrid(
                src,
                dst,
                src_coords_regrid,
                dst_coords_regrid,
                iter_regrid,
                self.use_relative_weights,
            )

        self._nd_regrid = nd_regrid

    def regrid(self, source, like, fill_value=np.nan):
        # Find coordinates that already match, and those that have to be
        # regridded, and those that exist in source but not in like (left
        # untouched)
        matching_dims, regrid_dims, add_dims = _match_dims(source, like)

        # Create tailor made regridding function: take method and ndims into
        # account and call it
        if self._first_call:

            if self.ndim_regrid is None:
                ndim_regrid = len(regrid_dims)
                if self.method == conductance and ndim_regrid > 2:
                    raise ValueError(
                        "The conductance method should not be applied to "
                        "regridding more than two dimensions"
                    )
                self.ndim_regrid = ndim_regrid

            self._make_regrid()
            self._first_call = False
        else:
            if not len(regrid_dims) == self.ndim_regrid:
                raise ValueError(
                    "Number of dimensions to regrid does not match: "
                    f"Regridder.ndim_regrid = {self.ndim_regrid}"
                )

        # Don't mutate source; src stands for source, dst for destination
        src = source.copy()

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

        # TODO: add methods for "conserve" and "linear"
        # Use xarray for nearest, and exit early.
        if self.method == "nearest":
            dst = xr.DataArray(
                data=source.reindex_like(like, method="nearest"),
                coords=dst_da_coords,
                dims=dst_dims,
            )
            return dst
        else:
            # Allocate dst
            # TODO: allocate lazy --> dask.array.full
            dst = xr.DataArray(
                data=np.full(dst_shape, fill_value), coords=dst_da_coords, dims=dst_dims
            )

        # TODO: check that axes are aligned
        dst_coords_regrid = [_coord(dst, dim) for dim in regrid_dims]
        src_coords_regrid = [_coord(src, dim) for dim in regrid_dims]
        # Transpose src so that dims to regrid are last
        src = src.transpose(*dst_dims)

        dst.values = self._nd_regrid(
            src.values, dst.values, src_coords_regrid, dst_coords_regrid
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        return np.nanmin(values)


def maximum(values, weights):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
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
