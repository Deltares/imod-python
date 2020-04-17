"""
Module that provides a class to do a variety of regridding operations, up to
three dimensions.

Before regridding, the dimension over which regridding should occur are
inferred, using the functions in the imod.prepare.common module. In case
multiple dimensions are represent, the data is reshaped such that a single loop
will regrid them all.

For example: let there be a DataArray with dimensions time, layer, y, and x. We
wish to regrid using an area weighted mean, over x and y. This means values
across times and layers are not aggregated together. In this case, the array is
reshaped into a 3D array, rather than a 4D array. Time and layer are stacked
into this first dimension together, so that a single loop suffices (see
common._reshape and _iter_regrid).

Functions can be incorporated into the multidimensional regridding. This is done
by making use of numba closures, since there's an overhead to passing function
objects directly. In this case, the function is simply compiled into the
specific regridding method, without additional overhead.

The regrid methods _regrid_{n}d are quite straightfoward. Using the indices that
and weights that have been gathered by _weights_1d, these methods fetch the
values from the source array (src), and pass it on to the aggregation method.
The single aggregated value is then filled into the destination array (dst).
"""
import warnings

import dask
import numba
import numpy as np
import xarray as xr

from imod.prepare import common, interpolate


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
    # k are indices of dst array
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

    # j, k are indices of dst array
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
    (
        ii,
        blocks_iz,
        blocks_weights_z,
        jj,
        blocks_iy,
        blocks_weights_y,
        kk,
        blocks_ix,
        blocks_weights_x,
    ) = inds_weights

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
        size, i_w = common._weights_1d(src_x, dst_x, use_relative_weights)
        for elem in i_w:
            inds_weights.append(elem)
        alloc_len *= size

    iter_src, iter_dst = common._reshape(src, dst, ndim_regrid)
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
        ``{"nearest", "multilinear", mean", "harmonic_mean", "geometric_mean",
        "sum", "minimum", "maximum", "mode", "median", "conductance"}``
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
    extra_overlap : integer, optional
        In case of chunked regridding, how many cells of additional overlap is
        necessary. Linear interpolation requires this for example, as it reaches
        beyond cell boundaries to compute values. Default value is 0.

    Examples
    --------
    Initialize the Regridder object:

    >>> mean_regridder = imod.prepare.Regridder(method="mean")

    Then call the ``regrid`` method to regrid.

    >>> result = mean_regridder.regrid(source, like)

    The regridder can be re-used if the number of regridding dimensions
    match, saving some time by not (re)compiling the regridding method.

    >>> second_result = mean_regridder.regrid(second_source, like)

    A one-liner is possible for single use:

    >>> result = imod.prepare.Regridder(method="mean").regrid(source, like)

    It's possible to provide your own methods to the ``Regridder``, provided that
    numba can compile them. They need to take the arguments ``values`` and
    ``weights``. Make sure they deal with ``nan`` values gracefully!

    >>> def p30(values, weights):
    >>>     return np.nanpercentile(values, 30)

    >>> p30_regridder = imod.prepare.Regridder(method=p30)
    >>> p30_result = p30_regridder.regrid(source, like)

    The Numba developers maintain a list of support Numpy features here:
    https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html

    In general, however, the provided methods should be adequate for your
    regridding needs.
    """

    def __init__(
        self, method, ndim_regrid=None, use_relative_weights=False, extra_overlap=0
    ):
        _method = common._get_method(method, common.METHODS)
        self.method = _method
        self.ndim_regrid = ndim_regrid
        self._first_call = True
        if _method == common.METHODS["conductance"]:
            use_relative_weights = True
        self.use_relative_weights = use_relative_weights
        if _method == common.METHODS["multilinear"]:
            extra_overlap = 1
        self.extra_overlap = extra_overlap

    def _make_regrid(self):
        iter_regrid = _make_regrid(self.method, self.ndim_regrid)
        iter_interp = interpolate._make_interp(self.ndim_regrid)

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
            return interpolate._nd_interp(
                src, dst, src_coords_regrid, dst_coords_regrid, iter_interp
            )

        if self.method == "nearest":
            pass
        elif self.method == "multilinear":
            self._nd_regrid = nd_interp
        else:
            self._nd_regrid = nd_regrid

    def _prepare(self, regrid_dims):
        # Create tailor made regridding function: take method and ndims into
        # account and call it
        if self._first_call:
            if self.ndim_regrid is None:
                ndim_regrid = len(regrid_dims)
                if self.method == common.METHODS["conductance"] and ndim_regrid > 2:
                    raise ValueError(
                        "The conductance method should not be applied to "
                        "regridding more than two dimensions"
                    )
                self.ndim_regrid = ndim_regrid

            self._make_regrid()
        else:
            if not len(regrid_dims) == self.ndim_regrid:
                raise ValueError(
                    "Number of dimensions to regrid does not match: "
                    f"Regridder.ndim_regrid = {self.ndim_regrid}"
                )

    def _regrid(self, src, like, fill_value):
        # Find coordinates that already match, and those that have to be
        # regridded, and those that exist in source but not in like (left
        # untouched)
        matching_dims, regrid_dims, add_dims = common._match_dims(src, like)

        # Order dimensions in the right way:
        # dimensions that are regridded end up at the end for efficient iteration
        dst_dims = (*add_dims, *matching_dims, *regrid_dims)
        dims_from_src = (*add_dims, *matching_dims)
        dims_from_like = tuple(regrid_dims)

        # Gather destination coordinates
        dst_da_coords, dst_shape = common._dst_coords(
            src, like, dims_from_src, dims_from_like
        )

        # Allocate dst
        dst = xr.DataArray(
            data=np.full(dst_shape, fill_value), coords=dst_da_coords, dims=dst_dims
        )
        # No overlap whatsoever, early exit
        if any(size == 0 for size in src.shape):
            return dst.values

        # TODO: check that axes are aligned
        dst_coords_regrid = [common._coord(dst, dim) for dim in regrid_dims]
        src_coords_regrid = [common._coord(src, dim) for dim in regrid_dims]
        # Transpose src so that dims to regrid are last
        src = src.transpose(*dst_dims)

        # Exit early if nothing is to be done
        if len(regrid_dims) == 0:
            return src.values.copy()
        else:
            dst.values = self._nd_regrid(
                src.values, dst.values, src_coords_regrid, dst_coords_regrid
            )
            return dst.values

    def _chunked_regrid(self, src, like, fill_value):
        like_expanded_slices, shape_chunks = common._define_slices(src, like)
        like_das = common._sel_chunks(like, like_expanded_slices)
        # Regridder should compute first chunk once
        # so numba has compiled the necessary functions for subsequent chunks
        if self._first_call:
            dst_da = like_das[0]
            matching_dims, regrid_dims, _ = common._match_dims(src, dst_da)
            chunk_src = common._slice_src(
                src, dst_da, matching_dims + regrid_dims, self.extra_overlap
            )
            a = self._regrid(chunk_src, dst_da, fill_value)
            arr1 = dask.array.from_array(a)
            self._first_call = False
        else:
            arr1 = None
        # At this point, the compiled method is part of the regridder class
        # and will be re-used by dask.

        np_collection = np.full(len(like_das), None)
        for i, dst_da in enumerate(like_das):
            # Skip first step if applicable
            if i == 0 and arr1 is not None:
                np_collection[0] = arr1
                continue
            matching_dims, regrid_dims, _ = common._match_dims(src, dst_da)

            # NOTA BENE: slice must occur BEFORE sending it to dask.delayed
            # if not, dask will attempt to allocate the full array!
            chunk_src = common._slice_src(
                src, dst_da, matching_dims + regrid_dims, self.extra_overlap
            )
            if any(
                size == 0 for size in chunk_src.shape
            ):  # zero overlap for the chunk, zero size chunk
                dask_array = dask.array.full(
                    shape=dst_da.shape, fill_value=fill_value, dtype=src.dtype
                )
            else:
                # Alllocation occurs inside
                result = dask.delayed(self._regrid, pure=True)(
                    chunk_src, dst_da, fill_value
                )
                dask_array = dask.array.from_delayed(
                    result, shape=dst_da.shape, dtype=src.dtype
                )

            np_collection[i] = dask_array

        # Determine the shape of the chunks, and reshape so dask.block does the right thing
        reshaped_collection = np.reshape(np_collection, shape_chunks).tolist()
        data = dask.array.block(reshaped_collection)
        return data

    def regrid(self, source, like, fill_value=np.nan):
        """
        Regrid ``source`` along dimensions that ``source`` and ``like`` share.
        These dimensions will be inferred the first time ``.regrid`` is called
        for the Regridder object.
        
        Following xarray conventions, nodata is assumed to ``np.nan``.
        
        Parameters
        ----------
        source : xr.DataArray of floats
        like : xr.DataArray of floats
            The like array present what the coordinates should look like.
        fill_value : float
            The fill_value. Defaults to np.nan
            
        Returns
        -------
        result : xr.DataArray
            Regridded result.
        """
        # Use xarray for nearest
        # TODO: replace by more efficient, specialized method
        if self.method == "nearest":
            matching_dims, regrid_dims, add_dims = common._match_dims(source, like)

            # Order dimensions in the right way:
            # dimensions that are regridded end up at the end for efficient iteration
            dst_dims = (*add_dims, *matching_dims, *regrid_dims)
            dims_from_src = (*add_dims, *matching_dims)
            dims_from_like = tuple(regrid_dims)

            # Gather destination coordinates
            dst_da_coords, _ = common._dst_coords(
                source, like, dims_from_src, dims_from_like
            )

            dst = source.reindex_like(like, method="nearest")
            dst = dst.assign_coords(dst_da_coords)
            return dst

        # Don't mutate source; src stands for source, dst for destination
        src = source.copy(deep=False)
        like = like.copy(deep=False)
        # TODO: Check dimensionality of coordinates
        # 2-d coordinates should raise a ValueError
        matching_dims, regrid_dims, add_dims = common._match_dims(src, like)
        # Exit early if nothing is to be done
        if len(regrid_dims) == 0:
            return src

        # Collect dimensions to flip to make everything ascending
        src, _ = common._increasing_dims(src, regrid_dims)
        like, flip_dst = common._increasing_dims(like, regrid_dims)
        # Ensure all dimensions have a dx coordinate, so that if the chunks
        # results in chunks which are size 1 along a dimension, the cellsize
        # can still be determined.
        src = common._set_cellsizes(src, regrid_dims)
        like = common._set_cellsizes(like, regrid_dims)

        # Prepare for regridding; quick checks
        self._prepare(regrid_dims)
        # Order dimensions in the right way:
        # dimensions that are regridded end up at the end for efficient iteration
        dst_dims = (*add_dims, *matching_dims, *regrid_dims)
        dims_from_src = (*add_dims, *matching_dims)
        dims_from_like = tuple(regrid_dims)
        # Gather destination coordinates
        dst_da_coords, _ = common._dst_coords(src, like, dims_from_src, dims_from_like)

        if src.chunks is None:
            self._first_call = False
            src = common._slice_src(
                src, like, matching_dims + regrid_dims, self.extra_overlap
            )
            data = self._regrid(src, like, fill_value)
        else:
            data = self._chunked_regrid(src, like, fill_value)

        dst = xr.DataArray(data=data, coords=dst_da_coords, dims=dst_dims)
        # Flip dimensions to return as like
        for dim in flip_dst:
            dst = dst.sel({dim: slice(None, None, -1)})
        # Transpose to original dimension coordinates
        # TODO: profile how much this matters!
        return dst.transpose(*source.dims)
