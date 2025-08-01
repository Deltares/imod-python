import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import imod
from imod.util.imports import MissingOptionalModule
from imod.visualize import common

try:
    import contextily as ctx
except ImportError:
    ctx = MissingOptionalModule("contextily")


def read_imod_legend(
    path: str | pathlib.Path,
) -> tuple[list[str], list[float], list[str]]:
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
    labels : List of strings of length N. These are the labels for the
        legend colors/classes.
    """

    # Read file. Do not rely the headers in the leg file.
    def _read(sep):
        return pd.read_csv(
            path,
            header=1,
            sep=sep,
            index_col=False,
            usecols=[0, 1, 2, 3, 4, 5],
            names=["upper", "lower", "red", "green", "blue", "labels"],
        )

    # Try both comma and whitespace separated
    try:
        legend = _read(sep=",")
    except ValueError:
        legend = _read(sep=r"\s+")

    # The colors in iMOD are formatted in RGB. Format to hexadecimal.
    red = legend["red"]
    blue = legend["blue"]
    green = legend["green"]
    colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in zip(red, green, blue)][::-1]
    levels = list(legend["lower"])[::-1][1:]
    labels = list(legend["labels"])[::-1]

    return colors, levels, labels


def _crs2string(crs):
    if isinstance(crs, str):
        if "epsg" in crs.lower():
            return crs
    try:
        return crs.to_string()
    except AttributeError:
        try:
            return crs["init"]
        except KeyError:
            return crs


def plot_map(
    raster,
    colors,
    levels,
    overlays=[],
    basemap=None,
    kwargs_raster=None,
    kwargs_colorbar=None,
    kwargs_basemap={},
    figsize=None,
    return_cbar=False,
    fig=None,
    ax=None,
):
    """
    Plot raster on a map with optional vector overlays and basemap.

    Parameters
    ----------
    raster : xr.DataArray
        2D grid to plot.
    colors : list of str, list of RGBA/RGBA tuples, colormap name (str), or
        LinearSegmentedColormap.

        If list, it should be a Matplotlib acceptable list of colors. Length N.
        Accepts both tuples of (R, G, B) and hexidecimal (e.g. `#7ec0ee`).

        If str, use an existing Matplotlib colormap. This function will
        autmatically add distinctive colors for pixels lower or high than the
        given min respectivly max level.

        If LinearSegmentedColormap, you can use something like
        `matplotlib.cm.get_cmap('jet')` as input. This function will not alter
        the colormap, so add under- and over-colors yourself.

        Looking for good colormaps? Try: http://colorbrewer2.org/ Choose a
        colormap, and use the HEX JS array.
    levels : listlike of floats or integers
        Boundaries between the legend colors/classes. Length: N - 1.
    overlays : list of dicts, optional
        Dicts contain geodataframe (key is "gdf"), and the keyword arguments for
        plotting the geodataframe.
    basemap : bool or contextily._providers.TileProvider, optional
        When `True` or a `contextily._providers.TileProvider` object: plot a
        basemap as a background for the plot and make the raster translucent. If
        `basemap=True`, then `CartoDB.Positron` is used as the default provider.
        If not set explicitly through kwargs_basemap, plot_map() will try and
        infer the crs from the raster or overlays, or fall back to EPSG:28992
        (Amersfoort/RDnew).

        *Requires contextily*

    kwargs_raster : dict of keyword arguments, optional
        These arguments are forwarded to ax.imshow()
    kwargs_colorbar : dict of keyword arguments, optional
        These arguments are forwarded to fig.colorbar(). The key label can be
        used to label the colorbar. Key whiten_triangles can be set to False to
        alter the default behavior of coloring the min / max triangles of the
        colorbar white if the value is not present in the map.
    kwargs_basemap : dict of keyword arguments, optional
        Except for "alpha", these arguments are forwarded to
        contextily.add_basemap(). Parameter "alpha" controls the transparency of
        raster.
    figsize : tuple of two floats or integers, optional
        This is used in plt.subplots(figsize)
    return_cbar : boolean, optional
        Return the matplotlib.Colorbar instance. Defaults to False.
    fig : matplotlib.figure, optional
        If provided, figure to which to add the map
    ax : matplot.ax, optional
        If provided, axis to which to add the map

    Returns
    -------
    fig : matplotlib.figure ax : matplotlib.ax if return_cbar == True: cbar :
    matplotlib.Colorbar

    Examples
    --------
    Plot with an overlay:

    >>> overlays = [{"gdf": geodataframe, "edgecolor": "black", "facecolor": "None"}]
    >>> imod.visualize.plot_map(raster, colors, levels, overlays)

    Label the colorbar:

    >>> imod.visualize.plot_map(raster, colors, levels, kwargs_colorbar={"label":"Head aquifer (m)"})

    Plot with a basemap:

    >>> import contextily as ctx
    >>> src = ctx.providers.Stamen.TonerLite
    >>> imod.visualize.plot_map(raster, colors, levels, basemap=src, kwargs_basemap={"alpha":0.6})

    """
    # Account for both None or False to skip adding a basemap
    if basemap is None or (isinstance(basemap, bool) and not basemap):
        add_basemap = False
    else:
        add_basemap = True

    # Read legend settings
    cmap, norm = common._cmapnorm_from_colorslevels(colors, levels)

    # Get extent
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial.spatial_reference(raster)

    # raster kwargs
    settings_raster = {"interpolation": "nearest", "extent": [xmin, xmax, ymin, ymax]}
    # if a basemap is added: set alpha of raster
    if add_basemap:
        settings_raster["alpha"] = kwargs_basemap.pop("alpha", 0.7)
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

    whiten_triangles = True
    if kwargs_colorbar is not None:
        whiten_triangles = kwargs_colorbar.pop("whiten_triangles", True)
        settings_cbar.update(kwargs_colorbar)

    # If not provided, make figure and axes
    # Catch case first where no figure provided, but ax was provided
    if fig is None and ax is not None:
        raise ValueError(
            "Axes provided, yet no figure is provided. Please provide a figure as well."
        )
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = plt.axes()

    # Make sure x is increasing, y is decreasing
    raster = raster.copy(deep=False)
    flip = slice(None, None, -1)
    if not raster.indexes["x"].is_monotonic_increasing:
        raster = raster.isel(x=flip)
    if not raster.indexes["y"].is_monotonic_decreasing:
        raster = raster.isel(y=flip)

    # Plot raster
    ax1 = ax.imshow(raster, cmap=cmap, norm=norm, zorder=1, **settings_raster)

    # Set ax imits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Make triangles white if data is not larger/smaller than legend_levels-range
    if whiten_triangles:
        if float(raster.max().compute()) < levels[-1]:
            ax1.cmap.set_over("#FFFFFF")
        if float(raster.min().compute()) > levels[0]:
            ax1.cmap.set_under("#FFFFFF")

    # Add colorbar
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    settings_cbar.pop("ticklabels", None)
    cbar = fig.colorbar(ax1, cax=cbar_ax, **settings_cbar)

    # Add overlays
    for i, overlay in enumerate(overlays):
        tmp = overlay.copy()
        gdf = tmp.pop("gdf")
        gdf.plot(ax=ax, zorder=2 + i, **tmp)

    # Add basemap, if basemap is neither None nor False
    if add_basemap:
        crs = "EPSG:28992"  # default Amersfoort/RDnew
        try:
            crs = _crs2string(kwargs_basemap.pop("crs"))
        except (KeyError, AttributeError):
            try:
                crs = _crs2string(raster.attrs["crs"])
            except (KeyError, AttributeError):
                for overlay in overlays:
                    if "crs" in overlay["gdf"]:
                        crs = _crs2string(overlay["gdf"].crs)
                        break

        if isinstance(basemap, bool):
            source = ctx.providers["CartoDB"]["Positron"]
        else:
            source = basemap

        ctx.add_basemap(ax=ax, source=source, crs=crs, **kwargs_basemap)

    # Return
    if return_cbar:
        return fig, ax, cbar
    else:
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
            if len(levels) > nlevels:
                # Decrease the number of levels
                # Pretty rough approach, but should be sufficient
                x = np.linspace(0.0, 100.0, nlevels)
                xp = np.linspace(0.0, 100.0, len(levels))
                yp = levels
                levels = np.interp(x, xp, yp)
            else:  # Can't make more levels out of only a few quantiles
                nlevels = len(levels)
        else:  # Go start to end
            vmin = float(a_yx.min())
            vmax = float(a_yx.max())
            levels = np.linspace(vmin, vmax, nlevels)
    elif isinstance(levels, (np.ndarray, list, tuple)):  # Pre-defined by user
        nlevels = len(levels)
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
    fig.colorbar(ax1, cax=cbar_ax)

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
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial.spatial_reference(da)
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
