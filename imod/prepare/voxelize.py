import numba
import numpy as np
import xarray as xr

from imod.prepare import common

# Voxelize does not support conductance method, nearest, or linear
METHODS = common.METHODS.copy()
METHODS.pop("conductance")
METHODS.pop("nearest")
METHODS.pop("multilinear")


@numba.njit(cache=True)
def _voxelize(src, dst, src_top, src_bot, dst_z, method):
    nlayer, nrow, ncol = src.shape
    nz = dst_z.size - 1
    values = np.zeros(nlayer)
    weights = np.zeros(nlayer)

    for i in range(nrow):
        for j in range(ncol):
            tops = src_top[:, i, j]
            bots = src_bot[:, i, j]

            # ii is index of dst
            for ii in range(nz):
                z0 = dst_z[ii]
                z1 = dst_z[ii + 1]
                if np.isnan(z0) or np.isnan(z1):
                    continue

                zb = min(z0, z1)
                zt = max(z0, z1)
                count = 0
                has_value = False
                # jj is index of src
                for jj in range(nlayer):
                    top = tops[jj]
                    bot = bots[jj]

                    overlap = common._overlap((bot, top), (zb, zt))
                    if overlap == 0:
                        continue

                    has_value = True
                    values[count] = src[jj, i, j]
                    weights[count] = overlap
                    count += 1
                else:
                    if has_value:
                        dst[ii, i, j] = method(values, weights)
                        # Reset
                        values[:count] = 0
                        weights[:count] = 0

    return dst


class Voxelizer:
    """
    Object to repeatedly voxelize similar objects. Compiles once on first call,
    can then be repeatedly called without JIT compilation overhead.

    Attributes
    ----------
    method : str, function
        The method to use for regridding. Default available methods are:
        ``{"mean", "harmonic_mean", "geometric_mean", "sum", "minimum",
        "maximum", "mode", "median", "max_overlap"}``

    Examples
    --------
    Usage is similar to the regridding. Initialize the Voxelizer object:

    >>> mean_voxelizer = imod.prepare.Voxelizer(method="mean")

    Then call the ``voxelize`` method to transform a layered dataset into a
    voxel based one. The vertical coordinates of the layers must be provided
    by ``top`` and ``bottom``.

    >>> mean_voxelizer.voxelize(source, top, bottom, like)

    If your data is already voxel based, i.e. the layers have tops and bottoms
    that do not differ with x or y, you should use a ``Regridder`` instead.

    It's possible to provide your own methods to the ``Regridder``, provided that
    numba can compile them. They need to take the arguments ``values`` and
    ``weights``. Make sure they deal with ``nan`` values gracefully!

    >>> def p30(values, weights):
    >>>     return np.nanpercentile(values, 30)

    >>> p30_voxelizer = imod.prepare.Voxelizer(method=p30)
    >>> p30_result = p30_voxelizer.regrid(source, top, bottom, like)

    The Numba developers maintain a list of support Numpy features here:
    https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html

    In general, however, the provided methods should be adequate for your
    voxelizing needs.
    """

    def __init__(self, method, use_relative_weights=False):
        _method = common._get_method(method, METHODS)
        self.method = _method
        self._first_call = True

    def _make_voxelize(self):
        """
        Use closure to avoid numba overhead
        """
        jit_method = numba.njit(self.method)

        @numba.njit
        def voxelize(src, dst, src_top, src_bot, dst_z):
            return _voxelize(src, dst, src_top, src_bot, dst_z, jit_method)

        self._voxelize = voxelize

    def voxelize(self, source, top, bottom, like):
        """

        Parameters
        ----------
        source : xr.DataArray
            The values of the layered model.
        top : xr.DataArray
            The vertical location of the layer tops.
        bottom : xr.DataArray
            The vertical location of the layer bottoms.
        like : xr.DataArray
            An example DataArray providing the coordinates of the voxelized
            results; what it should look like in terms of dimensions, data type,
            and coordinates.

        Returns
        -------
        voxelized : xr.DataArray
        """

        def dim_format(dims):
            return ", ".join(dim for dim in dims)

        # Checks on inputs
        if "z" not in like.dims:
            # might be a coordinate
            if "layer" in like.dims:
                if not like.coords["z"].dims == ("layer",):
                    raise ValueError('"z" has to be given in ``like`` coordinates')
        if "dz" not in like.coords:
            dzs = np.diff(like.coords["z"].values)
            dz = dzs[0]
            if not np.allclose(dzs, dz):
                raise ValueError(
                    '"dz" has to be given as a coordinate in case of'
                    ' non-equidistant "z" coordinate.'
                )
            like["dz"] = dz
        for da in [top, bottom, source]:
            if not da.dims == ("layer", "y", "x"):
                raise ValueError(
                    "Dimensions for top, bottom, and source have to be exactly"
                    f' ("layer", "y", "x"). Got instead {dim_format(da.dims)}.'
                )
        for da in [bottom, source]:
            for dim in ["layer", "y", "x"]:
                if not top[dim].equals(da[dim]):
                    raise ValueError(f"Input coordinates do not match along {dim}")

        if self._first_call:
            self._make_voxelize()
            self._first_call = False

        like_z = like["z"]
        if not like_z.indexes["z"].is_monotonic_increasing:
            like_z = like_z.isel(z=slice(None, None, -1))
            dst_z = common._coord(like_z, "z")[::-1]
        else:
            dst_z = common._coord(like_z, "z")

        dst_nlayer = like["z"].size
        _, nrow, ncol = source.shape

        dst_coords = {
            "z": like.coords["z"],
            "y": source.coords["y"],
            "x": source.coords["x"],
        }
        dst_dims = ("z", "y", "x")
        dst_shape = (dst_nlayer, nrow, ncol)

        dst = xr.DataArray(np.full(dst_shape, np.nan), dst_coords, dst_dims)
        dst.values = self._voxelize(
            source.values, dst.values, top.values, bottom.values, dst_z
        )

        return dst
