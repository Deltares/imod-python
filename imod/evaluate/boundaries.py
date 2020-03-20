import numba
import numpy as np
import xarray as xr


@numba.njit
def _interpolate_value_boundaries(values, z, threshold, out):
    nlay, nrow, ncol = values.shape
    for i in range(nrow):
        for j in range(ncol):
            n = 0
            if values[0, i, j] >= threshold:  # top cell exceeds threshold
                out[n, i, j] = z[0, i, j]
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
    present in the 3D values DataArray. Even boundaries represent exceedance boundaries, 
    at odd boundaries the values falls below the threshold again, as calculated from the top downward.

    Parameters
    ----------
    values : 3D xr.DataArray
        The datarray containing the values to search for boundaries. Dimensions ``layer``, ``y``, ``x``
    z : 3D xr.DataArray
        Datarray containing z-coordinates of cell midpoints. Dimensions ``layer``, ``y``, ``x``
    threshold : float
        Value threshold

    Returns
    -------
    xr.DataArray
        Z locations of exceedance and non-exceedance of threshold. Dimensions ``boundary``, ``y``, ``x`` 
    """
    values = values.load()
    out = xr.full_like(values, np.nan)
    _interpolate_value_boundaries(values.values, z.values, threshold, out.values)
    out = out.rename({"layer": "boundary"})
    return out.dropna(dim="boundary", how="all")
