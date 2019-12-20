import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import imod


def cross_section(
    da, colors, levels, layers=False, kwargs_pcolormesh={}, kwargs_colorbar={}
):
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
    colors : list of str, or list of RGB tuples
        Matplotlib acceptable list of colors. Length N.
        Accepts both tuples of (R, G, B) and hexidecimal (e.g. "#7ec0ee").

        Looking for good colormaps? Try: http://colorbrewer2.org/
        Choose a colormap, and use the HEX JS array.
    levels : listlike of floats or integers
        Boundaries between the legend colors/classes. Length: N - 1.
    layers : boolean, optional
        Whether to draw lines separating the layers.
    kwargs_pcolormesh
        Other optional keyword arguments for matplotlib.pcolormesh.
    kwargs_colorbar
        These arguments are forwarded to fig.colorbar()

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

    ncolors = len(colors)
    nlevels = len(levels)
    if not nlevels == ncolors - 1:
        raise ValueError(
            f"Incorrect number of levels. Number of colors is {ncolors},"
            f" expected {ncolors - 1}, got {nlevels} instead."
        )
    # Read legend settings
    cmap = matplotlib.colors.ListedColormap(colors[1:-1])
    cmap.set_under(colors[0])  # this is the color for values smaller than raster.min()
    cmap.set_over(colors[-1])  # this is the color for values larger than raster.max()
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)

    # cbar kwargs
    settings_cbar = {"ticks": levels, "extend": "both"}

    # Find a unit in the raster to use in the colorbar label
    try:
        settings_cbar["label"] = da.attrs["units"]
    except (KeyError, AttributeError):
        try:
            settings_cbar["label"] = da.attrs["unit"]
        except (KeyError, AttributeError):
            pass

    if kwargs_colorbar is not None:
        settings_cbar.update(kwargs_colorbar)

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
        X = (xmin + dx[0]) + dx[1:].cumsum()

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
    # Plot raster
    ax1 = ax.pcolormesh(X, Y, C, **kwargs_pcolormesh)
    if layers:
        Y[nodata] = np.nan
        for y in Y:
            ax.step(x=X[0], y=y)

    # Make triangles white if data is not larger/smaller than legend_levels-range
    if float(da.max().compute()) < levels[-1]:
        ax1.cmap.set_over("#FFFFFF")
    if float(da.min().compute()) > levels[0]:
        ax1.cmap.set_under("#FFFFFF")

    # Add colorbar
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(ax1, cmap=cmap, norm=norm, cax=cbar_ax, **settings_cbar)

    return fig, ax
