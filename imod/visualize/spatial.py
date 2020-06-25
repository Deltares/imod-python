import copy
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

import imod

# since geopandas is a big dependency that is sometimes hard to install
# and not always required, we made this an optional dependency
try:
    import geopandas as gpd
except ImportError:
    pass


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
    colors : list of str, list of RGBA/RGBA tuples, colormap name (str), or
	    LinearSegmentedColormap
        If list, it should be a Matplotlib acceptable list of colors. Length N.
        Accepts both tuples of (R, G, B) and hexidecimal (e.g. `#7ec0ee`).
		If str, use an existing Matplotlib colormap. This function will
		autmatically add distinctive colors for pixels lower or high than the given
		min respectivly max level.
		If LinearSegmentedColormap, you can use something like
		`matplotlib.cm.get_cmap('jet')` as input. This function will not alter
		the colormap, so add under- and over-colors yourself.

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
    # Read legend settings
    if isinstance(colors, matplotlib.colors.LinearSegmentedColormap):
        # use given cmap
        cmap = colors
    else:
        nlevels = len(levels)
        if isinstance(colors, str):
            # Use given cmap, but fix the under and over colors
            # The colormap (probably) does not have a nice under and over color.
            # So we cant use `cmap = matplotlib.cm.get_cmap(colors)`
            cmap = matplotlib.cm.get_cmap(colors)
            colors = cmap(np.linspace(0, 1, nlevels + 1))
        # Validate number of colors vs number of levels
        ncolors = len(colors)
        if not nlevels == ncolors - 1:
            raise ValueError(
                f"Incorrect number of levels. Number of colors is {ncolors},"
                f" expected {ncolors - 1} levels, got {nlevels} levels instead."
            )
        # Crate cmap from given list of colors
        cmap = matplotlib.colors.ListedColormap(colors[1:-1])
        cmap.set_under(
            colors[0]
        )  # this is the color for values smaller than raster.min()
        cmap.set_over(
            colors[-1]
        )  # this is the color for values larger than raster.max()
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)

    # Get extent
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(raster)

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

    # Make sure x is increasing, y is decreasing
    raster = raster.copy(deep=False)
    flip = slice(None, None, -1)
    if not raster.indexes["x"].is_monotonic_increasing:
        raster = raster.isel(x=flip)
    if not raster.indexes["y"].is_monotonic_decreasing:
        raster = raster.isel(y=flip)

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


def _colorscale(a_yx, levels, cmap, quantile_colorscale):
    """
    This is an attempt to automatically create somewhat robust color scales.

    Parameters
    ----------
    a_yx : xr.DataArray
        2D DataArray with only dimensions ("y", "x")
    levels : integer, np.ndarray or None
        Number of levels (if integer), or level boundaries (if ndarray)
    quantile_colorscale : boolean
        Whether to create a colorscale based on quantile classification

    Returns
    -------
    norm : matplotlib.colors.BoundaryNorm
    cmap : matplotlib.colors.ListedColormap
    """
    # This is all an attempt at a somewhat robust colorscale handling
    if levels is None:  # Nothing given, default to 25 colors
        levels = 25
    if isinstance(levels, int):
        nlevels = levels
        if quantile_colorscale:
            levels = np.unique(np.nanpercentile(a_yx.values, np.linspace(0, 100, 101)))
            if levels.size > nlevels:
                # Decrease the number of levels
                # Pretty rough approach, but should be sufficient
                x = np.linspace(0.0, 100.0, nlevels)
                xp = np.linspace(0.0, 100.0, levels.size)
                yp = levels
                levels = np.interp(x, xp, yp)
            else:  # Can't make more levels out of only a few quantiles
                nlevels = levels.size
        else:  # Go start to end
            vmin = float(a_yx.min())
            vmax = float(a_yx.max())
            levels = np.linspace(vmin, vmax, nlevels)
    elif isinstance(levels, (np.ndarray, list, tuple)):  # Pre-defined by user
        nlevels = levels.size
    else:
        raise ValueError("levels argument should be None, an integer, or an array.")

    if nlevels < 3:  # let matplotlib take care of it
        norm = None
    else:
        norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=nlevels)

    # Interpolate colormap to nlevels
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)
    # cmap is a callable object
    cmap = matplotlib.colors.ListedColormap(cmap(np.linspace(0.0, 1.0, nlevels)))

    return norm, cmap


def _imshow_xy(
    a_yx, fname, title, cmap, overlays, quantile_colorscale, figsize, settings, levels
):
    fig, ax = plt.subplots(figsize=figsize)
    norm, cmap = _colorscale(a_yx, levels, cmap, quantile_colorscale)
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
        return time.strftime("%Y%m%d%H%M%S")


def imshow_topview(
    da,
    name,
    directory=".",
    cmap="viridis",
    overlays=[],
    quantile_colorscale=True,
    figsize=(8, 8),
    levels=None,
):
    """
    Automatically colors by quantile.

    Dumps PNGs into directory of choice.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if "x" not in da.dims or "y" not in da.dims:
        raise ValueError("DataArray must have dims x and y.")
    directory = pathlib.Path(directory)
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(da)
    settings = {"interpolation": "nearest", "extent": [xmin, xmax, ymin, ymax]}
    extradims = [dim for dim in da.dims if dim not in ("x", "y")]

    if len(extradims) == 0:
        fname = directory / f"{name}.png"
        _imshow_xy(
            da,
            fname,
            name,
            cmap,
            overlays,
            quantile_colorscale,
            figsize,
            settings,
            levels,
        )
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
            _imshow_xy(
                a_yx,
                fname,
                title,
                cmap,
                overlays,
                quantile_colorscale,
                figsize,
                settings,
                levels,
            )
