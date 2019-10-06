import matplotlib.pyplot as plt
import numpy as np

import imod


def cross_section(da, layers=False, ax=None, **kwargs):
    """
    Wraps matplotlib.pcolormesh to draw cross-sections, drawing cell boundaries
    accurately.

    Parameters
    ----------
    da : xr.DataArray
        Two dimensional DataArray containing data of the cross section. The
        first dimension must be "layer", and the second dimension will be used
        as the x-axis for the cross-section.
        Coordinates "top" and "bottom" must be present.
    layers : boolean, optional
        Whether to draw lines separating the layers.
    ax : matplotlib.ax, optional
        Matplotlib axes object on which the cross-section will be drawn.
    **kwargs
        Other optional keyword argument for matplotlib.pcolormesh.

    Returns
    -------
    None    
    """
    data = da.values
    xcoord = da.dims[1]
    dx, xmin, xmax = imod.util.coord_reference(da[xcoord])
    if dx < 0.0:
        dx = abs(dx)
        data = data[:, ::-1]

    Y = np.vstack([da["top"].isel(layer=0).values, da["bottom"].values])
    Y = np.repeat(Y, 2, 1)
    nodata = np.isnan(Y)
    Y[nodata] = 0.0

    X = np.arange(xmin + dx, xmax, dx)
    X = np.hstack([xmin, np.repeat(X, 2), xmax])
    X = np.full_like(Y, X)

    nrow, ncol = Y.shape
    C = np.full((nrow - 1, ncol - 1), np.nan)
    C[:, 0::2] = data

    if ax is not None:
        ax.pcolormesh(X, Y, C, **kwargs)
        if layers:
            Y[nodata] = np.nan
            for y in Y:
                ax.step(x=X[0], y=y)
    else:
        plt.pcolormesh(X, Y, C, **kwargs)
        if layers:
            Y[nodata] = np.nan
            for y in Y:
                plt.step(x=X[0], y=y)
