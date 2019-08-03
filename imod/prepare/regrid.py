import numba
import numpy as np
import xarray as xr


from .common import _check_monotonic
from .common import _match_dims
from .common import _slice_src
from .common import _dst_coords
from .common import _check_monotonic
from .common import _coord
from .common import _get_method
from .common import _overlap
from .common import _is_increasing
from .common import _reshape
from .common import METHODS
from .interpolate import _nd_interp, _make_interp, _iter_interp


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
        `ndim_regrid` will be inferred. It serves to prevent regridding over an
        unexpected number of dimensions; say you want to regrid over only two
        dimensions. Due to an input error in the coordinates of `like`, three
        dimensions may be inferred in the first `.regrid` call. An error will
        be raised if ndim_regrid not match the number of inferred dimensions.
        Default value is None.
    use_relative_weights : bool, optional
        Whether to use relative weights in the regridding method or not.
        Relative weights are defined as: cell_overlap / source_cellsize, for
        every axis.

        This argument should only be used if you are providing your own
        `method` as a function, where the function requires relative, rather
        than absolute weights (the provided `conductance` method requires
        relative weights, for example). Default value is False.

    Examples
    --------
    Initialize the Regridder object:

    >>> mean_regridder = imod.prepare.Regridder(method="mean")

    Then call the `regrid` method to regrid.

    >>> result = mean_regridder(source, like)

    The regridder can be re-used if the number of regridding dimensions
    match, saving some time by not (re)compiling the regridding method.

    >>> second_result = mean_regrid(second_source, like)

    A one-liner is possible for single use:

    >>> result = imod.prepare.Regridder(method="mean").regrid(source, like)

    It's possible to provide your own methods to the `Regridder`, provided that
    numba can compile them. They need to take the arguments `values` and
    `weights`. Make sure they deal with nan values gracefully!

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
        if _method == METHODS["conductance"]:
            use_relative_weights = True
        self.use_relative_weights = use_relative_weights

    def _make_regrid(self):
        iter_regrid = _make_regrid(self.method, self.ndim_regrid)
        iter_interp = _make_interp(self.ndim_regrid)

        def nd_regrid(src, dst, src_coords_regrid, dst_coords_regrid):
            return _nd_regrid(
                src,
                dst,
                src_coords_regrid,
                dst_coords_regrid,
                iter_regrid,
                self.use_relative_weights,
            )

        def nd_interp(src, dst, src_coords_regrid, dst_coords_regrid):
            return _nd_interp(
                src, dst, src_coords_regrid, dst_coords_regrid, iter_interp
            )

        if self.method == "nearest":
            pass
        elif self.method == "multilinear":
            self._nd_regrid = nd_interp
        else:
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
                if self.method == METHODS["conductance"] and ndim_regrid > 2:
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
