import numba
import numpy as np
import xarray as xr


@numba.njit
def _interpolate_value_boundaries_z1d(values, z, dz_top, threshold, out):
    nlay, nrow, ncol = values.shape
    for i in range(nrow):
        for j in range(ncol):
            n = 0
            if values[0, i, j] >= threshold:  # top cell exceeds threshold
                out[n, i, j] = z[0] + 0.5 * dz_top
                n += 1
            for k in range(1, nlay):
                # from top downward
                if (n % 2 == 0 and values[k, i, j] >= threshold) or (
                    n % 2 == 1 and values[k, i, j] <= threshold
                ):  # exceeding (n even) or falling below threshold (n odd)
                    # interpolate z coord of threshold
                    out[n, i, j] = z[k - 1] + (threshold - values[k - 1, i, j]) * (
                        z[k] - z[k - 1]
                    ) / (values[k, i, j] - values[k - 1, i, j])
                    n += 1


@numba.njit
def _interpolate_value_boundaries_z3d(values, z, dz_top, threshold, out):
    nlay, nrow, ncol = values.shape
    for i in range(nrow):
        for j in range(ncol):
            n = 0
            if values[0, i, j] >= threshold:  # top cell exceeds threshold
                out[n, i, j] = z[0, i, j] + 0.5 * dz_top[i, j]
                n += 1
            for k in range(1, nlay):
                # from top downward
                if (n % 2 == 0 and values[k, i, j] >= threshold) or (
                    n % 2 == 1 and values[k, i, j] <= threshold
                ):  # exceeding (n even) or falling below threshold (n odd)
                    # interpolate z coord of threshold
                    out[n, i, j] = z[k - 1, i, j] + (
                        threshold - values[k - 1, i, j]
                    ) * (z[k, i, j] - z[k - 1, i, j]) / (
                        values[k, i, j] - values[k - 1, i, j]
                    )
                    n += 1


def interpolate_value_boundaries(values, z, threshold):
    """Function that returns all exceedance and non-exceedance boundaries for
    a given threshold in a 3D values DataArray. Returned z-coordinates are 
    linearly interpolated between cell mids. As many boundaries are returned as are maximally 
    present in the 3D values DataArray. Function returns xr.DataArray of exceedance boundaries 
    and xr.DataArray of z-coordinates where values fall below the set treshold.

    Parameters
    ----------
    values : 3D xr.DataArray
        The datarray containing the values to search for boundaries. Dimensions ``layer``, ``y``, ``x``
    z : 1D or 3D xr.DataArray
        Datarray containing z-coordinates of cell midpoints. Dimensions ``layer``, ``y``, ``x``. Should contain a dz coordinate.
    threshold : float
        Value threshold

    Returns
    -------
    xr.DataArray
        Z locations of successive exceedances of threshold from the top down. Dimensions ``boundary``, ``y``, ``x`` 
    xr.DataArray
        Z locations of successive instances of falling below threshold from the top down. Dimensions ``boundary``, ``y``, ``x`` 
    """

    if "dz" not in z.coords:
        raise ValueError('Dataarray "z" must contain a "dz" coordinate')

    values = values.load()
    out = xr.full_like(values, np.nan)
    if len(z.dims) == 1:
        _interpolate_value_boundaries_z1d(
            values.values, z.values, z.dz.values[0], threshold, out.values
        )
    else:
        if len(z.dz.dims) == 1:
            # broadcast to 2d
            dz_top = xr.ones_like(z.isel(layer=0)) * z.dz.isel(layer=0)
        else:
            dz_top = z.dz.isel(layer=0)
        _interpolate_value_boundaries_z3d(
            values.values, z.values, dz_top.values, threshold, out.values
        )
    out = out.rename({"layer": "boundary"})
    out = out.dropna(dim="boundary", how="all")

    # _interpolate_value_boundaries returns exceedance / falling below as even- and odd-indexed boundaries
    # separate and renumber them for convenience
    out.coords["boundary"] = np.arange(len(out.coords["boundary"]))
    exceedance = out.sel(boundary=slice(0, len(out.coords["boundary"]), 2))
    exceedance.coords["boundary"] = np.arange(len(exceedance.coords["boundary"]))
    fallbelow = out.sel(boundary=slice(1, len(out.coords["boundary"]), 2))
    fallbelow.coords["boundary"] = np.arange(len(fallbelow.coords["boundary"]))

    return exceedance, fallbelow
