import numpy as np
import numba
import xarray as xr


from imod.prepare import common


# LayerRegrid does not support conductance method, nearest, or linear
METHODS = common.METHODS.copy()
METHODS.pop("conductance")
METHODS.pop("nearest")
METHODS.pop("multilinear")


@numba.njit(cache=True)
def _regrid_layers(
    src, dst, src_top, dst_top, src_bot, dst_bot, values, weights, method
):
    """
    Maps one set of layers unto the other.
    """
    nlayer_src, nrow, ncol = src.shape
    nlayer_dst = dst.shape[0]
    values = np.zeros(nlayer_src)
    weights = np.zeros(nlayer_src)

    for i in range(nrow):
        for j in range(ncol):
            src_t = src_top[:, i, j]
            dst_t = dst_top[:, i, j]
            src_b = src_bot[:, i, j]
            dst_b = dst_bot[:, i, j]

            # ii is index of dst
            for ii in range(nlayer_dst):
                dt = dst_t[ii]
                db = dst_b[ii]
                if np.isnan(dt) or np.isnan(db):
                    continue

                count = 0
                has_value = False
                # jj is index of src
                for jj in range(nlayer_src):
                    st = src_t[jj]
                    sb = src_b[jj]

                    overlap = common._overlap((db, dt), (sb, st))
                    if overlap == 0:
                        continue

                    has_value = True
                    values[count] = src[jj, i, j]
                    values[count] = overlap
                    count += 1
                else:
                    if has_value:
                        dst[ii, i, j] = method(values, weights)
                        # Reset
                        values[:count] = 0
                        weights[:count] = 0

    return dst


class LayerRegridder:
    """
    Object to repeatedly voxelize similar objects. Compiles once on first call,
    can then be repeatedly called without JIT compilation overhead.

    Attributes
    ----------
    method : str, function
        The method to use for regridding. Default available methods are:
        ``{"mean", "harmonic_mean", "geometric_mean", "sum", "minimum",
        "maximum", "mode", "median", "max_overlap"}``
    """

    def __init__(self, method):
        _method = common._get_method(method, METHODS)
        self.method = _method
        self.first_call = True

    def _make_regrid(self):
        """
        Use closure to avoid numba overhead
        """
        jit_method = numba.njit(self.method)

        @numba.njit
        def regrid(src, dst, src_top, dst_top, src_bot, dst_bot, weights, values):
            return _regrid_layers(
                src, dst, src_top, dst_top, src_bot, dst_bot, jit_method
            )

        self._regrid = regrid

    def regrid(
        self, source, source_top, source_bottom, destination_top, destination_bottom
    ):
        """
        Parameters
        ----------
        source : xr.DataArray
            The values of the layered model.
        source_top : xr.DataArray
            The vertical location of the layer tops.
        destination_top : xr.DataArray
            The vertical location of the layer tops.
        source_bottom : xr.DataArray
            The vertical location of the layer bottoms.
        destination_bottom : xr.DataArray
            The vertical location of the layer bottoms.

        Returns
        -------
        regridded : xr.DataArray
        """

        def dim_format(dims):
            return ", ".join(dim for dim in dims)

        # Checks on inputs
        for da in [
            source_top,
            source_bottom,
            source,
            destination_bottom,
            destination_top,
        ]:
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
            self._make_regrid()
            self._first_call = False

        dst = xr.full_like(destination_top, np.nan, dtype=source.dtype)
        dst.values = self._regrid(
            source.values,
            dst.values,
            source_top.values,
            destination_top.values,
            source_bottom.values,
            destination_bottom.values,
        )
        return dst
