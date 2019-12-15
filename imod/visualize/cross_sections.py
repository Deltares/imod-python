import matplotlib.pyplot as plt
import numpy as np

import imod


def cross_section(da, layers=False, **kwargs):
    """
    Wraps matplotlib.pcolormesh to draw cross-sections, drawing cell boundaries
    accurately.

    Parameters
    ----------
    da : xr.DataArray
        Two dimensional DataArray containing data of the cross section. One
        dimension must be "layer", and the second dimension will be used as the
        x-axis for the cross-section.

        Coordinates "top" and "bottom" must be present, and must have the same
        dimensions.
    layers : boolean, optional
        Whether to draw lines separating the layers.
    **kwargs
        Other optional keyword arguments for matplotlib.pcolormesh.

    Returns
    -------
    fig : matplotlib.figure
    ax : matplotlig.ax
    """
    if len(da.dims) != 2:
        raise ValueError("DataArray must be 2D")
    if "layer" not in da.dims:
        raise ValueError('DataArray must contain dimension "layer"')
    if "top" not in da.coords:
        raise ValueError('DataArray must contain coordinate "top"')
    if "bottom" not in da.coords:
        raise ValueError('DataArray must contain coordinate "bottom"')
    if len(da["top"].dims) != 2:
        raise ValueError('"top" coordinate be 2D')
    if len(da["bottom"].dims) != 2:
        raise ValueError('"bottom" coordinate be 2D')

    dims = tuple(da.dims)
    if not dims[0] == "layer":
        # Switch 'm around
        dims = (dims[1], dims[0])

    # Ensure dimensions are in the right order
    da = da.transpose(*dims, transpose_coords=True)
    data = da.values
    xcoord = da.dims[1]
    dx, xmin, xmax = imod.util.coord_reference(da[xcoord])
    if isinstance(dx, (int, float)):
        if dx < 0.0:
            dx = abs(dx)
            data = data[:, ::-1]
        X = np.arange(xmin + dx, xmax, dx)
    else:  # assuming dx is an array, non-equidistant dx
        if (dx < 0.0).all():
            dx = abs(dx)
            data = data[:, ::-1]
        elif (dx > 0.0).all():
            pass
        else:
            raise ValueError(f"{xcoord} is not monotonic")
        X = (xmin + dx[0]) + dx[1:]

    Y = np.vstack([da["top"].isel(layer=0).values, da["bottom"].values])
    Y = np.repeat(Y, 2, 1)
    nodata = np.isnan(Y)
    Y[nodata] = 0.0

    X = np.hstack([xmin, np.repeat(X, 2), xmax])
    X = np.full_like(Y, X)

    nrow, ncol = Y.shape
    C = np.full((nrow - 1, ncol - 1), np.nan)
    C[:, 0::2] = data

    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, C, **kwargs)
    if layers:
        Y[nodata] = np.nan
        for y in Y:
            ax.step(x=X[0], y=y)
    return fig, ax
