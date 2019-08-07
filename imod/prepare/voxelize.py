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
def _voxelize(src, dst, src_top, src_bot, dst_z, weights, values, method):
    nlayer, nrow, ncol = src.shape
    nz = dst_z.size - 1

    for i in range(nrow):
        for j in range(ncol):
            tops = src_top[:, i, j]
            bots = src_bot[:, i, j]

            # ii is index of dst
            for ii in range(nz):
                z0 = dst_z[ii]
                z1 = dst_z[ii + 1]
                zb = min(z0, z1)
                zt = max(z0, z1)

                count = 0
                has_value = False
                # jj is index of src
                for jj in range(nlayer):
                    top = tops[jj]
                    bot = bots[jj]
                    if np.isnan(z0) or np.isnan(z1):
                        continue

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
        {"mean", "harmonic_mean", "geometric_mean", "sum", "minimum",
        "maximum", "mode", "median", "max_overlap"}
    """

    def __init__(self, method, use_relative_weights=False):
        _method = common._get_method(method, METHODS)
        self.method = _method
        self._first_call = True

    def _make_voxelize(self):
        """
        Use closure to avoid numba overhead
        """
        jit_method = numba.njit(self.method, cache=True)

        @numba.njit
        def voxelize(src, dst, src_top, src_bot, dst_z, weights, values):
            return _voxelize(
                src, dst, src_top, src_bot, dst_z, weights, values, jit_method
            )

        self._voxelize = voxelize

    def voxelize(self, source, top, bottom, like):
        def dim_format(dims):
            return ", ".join(dim for dim in dims)

        # Checks on inputs
        if not "z" in like.dims:
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
            for (k1, v1), (_, v2) in zip(top.coords.items(), da.coords.items()):
                if not v1.equals(v2):
                    raise ValueError(f"Input coordinates do not match along {k1}")

        if self._first_call:
            self._make_voxelize()
            self._first_call = False

        dst_nlayer = like["z"].size
        dst_z = common._coord(like, "z")
        _, nrow, ncol = source.shape

        src_max_thickness = float((top - bottom).max())
        dst_min_dz = np.abs(np.diff(dst_z)).min()
        alloc_len = int(np.ceil(src_max_thickness / dst_min_dz))
        values = np.zeros(alloc_len)
        weights = np.zeros(alloc_len)

        dst_coords = {
            "z": like.coords["z"],
            "y": source.coords["y"],
            "x": source.coords["x"],
        }
        dst_dims = ("z", "y", "x")
        dst_shape = (dst_nlayer, nrow, ncol)

        dst = xr.DataArray(np.full(dst_shape, np.nan), dst_coords, dst_dims)
        dst.values = self._voxelize(
            source.values, dst.values, top.values, bottom.values, dst_z, weights, values
        )

        return dst
