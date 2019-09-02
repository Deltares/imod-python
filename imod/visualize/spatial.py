import copy
import pathlib

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

# since geopandas is a big dependency that is sometimes hard to install
# and not always required, we made this an optional dependency
try:
    import geopandas as gpd
except ImportError:
    pass

import imod


def read_imod_legend(path):
    """
    Parameters
    ----------
    path : str
        Path to iMOD .leg file.

    Returns
    -------
    colors : List of hex colors of length N.
    levels : List of floats of length N-1. These are the boundaries between
        the legend colors/classes.
    """

    # Read file. Do not rely the headers in the leg file.
    def _read(delim_whitespace):
        return pd.read_csv(
            path,
            header=1,
            delim_whitespace=delim_whitespace,
            index_col=False,
            usecols=[0, 1, 2, 3, 4],
            names=["upper", "lower", "red", "green", "blue"],
        )

    # Try both comma and whitespace separated
    try:
        legend = _read(delim_whitespace=False)
    except:
        legend = _read(delim_whitespace=True)

    # The colors in iMOD are formatted in RGB. Format to hexadecimal.
    red = legend["red"]
    blue = legend["blue"]
    green = legend["green"]
    colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in zip(red, green, blue)][::-1]
    levels = list(legend["lower"])[::-1][1:]

    return colors, levels


def plot_map(
    raster,
    colors,
    levels,
    overlays=[],
    kwargs_raster=None,
    kwargs_colorbar=None,
    figsize=None,
):
    """
    Parameters
    ----------
    raster : xr.DataArray
        2D grid to plot.
    colors : list of str, or list of RGB tuples
        Matplotlib acceptable list of colors. Length N.
        Accepts both tuples of (R, G, B) and hexidecimal (e.g. "#7ec0ee").

        Looking for good colormaps? Try: http://colorbrewer2.org/
        Choose a colormap, and use the HEX JS array.
    levels : listlike of floats or integers
        Boundaries between the legend colors/classes. Length: N - 1.
    overlays : list of dicts, optional
        Dicts contain geodataframe (key is "gdf"), and the keyword arguments
        for plotting the geodataframe.
    kwargs_raster : dict of keyword arguments, optional
        These arguments are forwarded to ax.imshow()
    kwargs_colorbar : dict of keyword arguments, optional
        These arguments are forwarded to fig.colorbar()
    figsize : tuple of two floats or integers, optional
        This is used in plt.subplots(figsize)

    Returns
    -------
    fig : matplotlib.figure
    ax : matplotlig.ax

    Examples
    --------
    Plot with an overlay:
    
    >>> overlays = [{"gdf": geodataframe, "edgecolor": "black"}]
    >>> imod.visualize.spatial.plot_map(raster, legend, overlays)
    """
    ncolors = len(colors)
    nlevels = len(levels)
    if not nlevels == ncolors - 1:
        raise ValueError(
            f"Incorrect number of levels. Number of colors is {ncolors},"
            f" expected {ncolors - 1}, got {nlevels} instead."
        )

    # Get extent
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(raster)

    # Read legend settings
    cmap = matplotlib.colors.ListedColormap(colors[1:-1])
    cmap.set_under(colors[0])  # this is the color for values smaller than raster.min()
    cmap.set_over(colors[-1])  # this is the color for values larger than raster.max()
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)

    # raster kwargs
    settings_raster = {"interpolation": "nearest", "extent": [xmin, xmax, ymin, ymax]}
    if kwargs_raster is not None:
        settings_raster.update(kwargs_raster)

    # cbar kwargs
    settings_cbar = {"ticks": levels, "extend": "both"}

    # Find a unit in the raster to use in the colorbar label
    try:
        settings_cbar["label"] = raster.attrs["units"]
    except (KeyError, AttributeError):
        try:
            settings_cbar["label"] = raster.attrs["unit"]
        except (KeyError, AttributeError):
            pass

    if kwargs_colorbar is not None:
        settings_cbar.update(kwargs_colorbar)

    # Make figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot raster
    ax1 = ax.imshow(raster, cmap=cmap, norm=norm, **settings_raster)

    # Set ax imits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Make triangles white if data is not larger/smaller than legend_levels-range
    if float(raster.max().compute()) < levels[-1]:
        ax1.cmap.set_over("#FFFFFF")
    if float(raster.min().compute()) > levels[0]:
        ax1.cmap.set_under("#FFFFFF")

    # Add colorbar
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(ax1, cmap=cmap, norm=norm, cax=cbar_ax, **settings_cbar)

    # Add overlays
    for overlay in overlays:
        tmp = overlay.copy()
        gdf = tmp.pop("gdf")
        gdf.plot(ax=ax, **tmp)

    # Return
    return fig, ax


def imshow(a_yx, fname, title, cmap, overlays, quantile_colorscale, figsize, settings):
    fig, ax = plt.subplots(figsize=figsize)
    if quantile_colorscale:
        levels = np.unique(np.percentile(a_yx.values, np.linspace(0, 100, 101)))
        if levels.size < 3:  # let matplotlib take care of it
            norm = None
        else:
            norm = matplotlib.colors.BoundaryNorm(levels, 256)
    else:
        norm = None
    ax1 = ax.imshow(a_yx, cmap=cmap, norm=norm, **settings)
    for overlay in overlays:
        tmp = overlay.copy()
        gdf = tmp.pop("gdf")
        gdf.plot(ax=ax, **tmp)

    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(ax1, cmap=cmap, norm=norm, cax=cbar_ax)

    ax.set_title(title)
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


def format_time(time):
    if isinstance(time, np.datetime64):
        # The following line is because numpy.datetime64[ns] does not
        # support converting to datetime, but returns an integer instead.
        # This solution is 20 times faster than using pd.to_datetime()
        return time.astype("datetime64[us]").item().strftime("%Y%m%d%H%M%S")
    else:
        return coordval.strftime("%Y%m%d%H%M%S")


def nd_imshow(
    da,
    name,
    directory=".",
    cmap="viridis",
    overlays=[],
    quantile_colorscale=True,
    figsize=(8, 8),
):
    """
    Automatically colors by quantile.

    Dumps PNGs into directory of choice.
    """
    if "x" not in da.dims or "y" not in da.dims:
        raise ValueError("DataArray must have dims x and y.")
    directory = pathlib.Path(directory)
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(da)
    settings = {"interpolation": "nearest", "extent": [xmin, xmax, ymin, ymax]}
    extradims = [dim for dim in da.dims if dim not in ("x", "y")]

    if len(extradims) == 0:
        fname = directory / f"{name}.png"
        imshow(da, fname, name, cmap, overlays, quantile_colorscale, figsize, settings)
    else:
        stacked = da.stack(idf=extradims)
        for coordvals, a_yx in list(stacked.groupby("idf")):
            if a_yx.isnull().all():
                continue

            fname_parts = []
            title_parts = []
            for key, coordval in zip(extradims, coordvals):
                title_parts.append(f"{key}: {coordval}")
                if key == "time":
                    coordval = format_time(coordval)
                fname_parts.append(f"{key}{coordval}")

            fname_parts = "_".join(fname_parts)
            title_parts = ", ".join(title_parts)
            fname = directory / f"{name}_{fname_parts}.png"
            title = f"{name}, {title_parts}"
            imshow(
                a_yx,
                fname,
                title,
                cmap,
                overlays,
                quantile_colorscale,
                figsize,
                settings,
            )
