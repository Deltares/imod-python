import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import geopandas as gpd
import pandas as pd
import copy
import imod


def read_imod_legend(path):
    """
    Parameters
    ----------
    path : str
        Path to iMod .leg file.

    Returns
    -------
    colors : List of hex colors of length N.
    levels : List of floats of length N-1. These are the boundaries between the legend colors/classes.
    """

    # Read file. Do not rely the headers in the leg file.
    leg = pd.read_csv(
        path,
        header=1,
        sep="[\s,]+",
        index_col=False,
        engine="python",
        usecols=[0, 1, 2, 3, 4],
        names=["UPPERBND", "LOWERBND", "IRED", "IGREEN", "IBLUE"],
    )

    # The colors in iMod are formatted in an ancient format. Reform to hex.
    leg.loc[:, "color"] = [
        "#%02x%02x%02x"
        % (leg.loc[i, "IRED"], leg.loc[i, "IGREEN"], leg.loc[i, "IBLUE"])
        for i, row in leg.iterrows()
    ]

    # Return colors, values/ticks
    return list(leg["color"])[::-1], list(leg["LOWERBND"][::-1])[1:]


def plot_map(
    raster,
    legend_colors,
    legend_levels,
    overlays=[],
    kwargs_raster=None,
    kwargs_cbar=None,
    figsize=None,
):
    """
    Parameters
    ----------
    raster : xr.DataArray
        2D grid to plot.
    legend_colors : list of str, or list of RGB tuples
        Matplotlib acceptable list of colors. Length N.
    legend_levels : listlike of floats or integers
        Boundaries between the legend colors/classes. Length: N-1.
    overlays : list of dicts
        Dicts contain geodataframe (key is "gdf"), and the keyword arguments for plotting the geodataframe.
    kwargs_raster : dict of keyword arguments
        These agruments are passed forward to for ax.imshow()
    kwargs_cbar : dict of keyword arguments
        These agruments are passed forward to for fig.colorbar()
    figsize : tuple of two floats or integers
        This is used in plt.subplots(figsize)
    
    Returns
    -------
    fig : matplotlib.figure
    ax : matplotlig.ax    
    """

    # Get extend
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(raster)

    # Read legend settings
    cmap = matplotlib.colors.ListedColormap(legend_colors[1:-1])
    cmap.set_under(
        legend_colors[0]
    )  # this is the color for values smaller than raster.min()
    cmap.set_over(
        legend_colors[-1]
    )  # this is the color for values larger than raster.max()
    norm = matplotlib.colors.BoundaryNorm(legend_levels, cmap.N)

    # raster kwargs
    settings_raster = {"interpolation": "nearest", "extent": [xmin, xmax, ymin, ymax]}
    if kwargs_raster is not None:
        settings_raster.update(kwargs_raster)

    # cbar kwargs
    settings_cbar = {"ticks": legend_levels, "extend": "both"}

    # Find a unit in the raster to use in the colorbar label
    try:
        settings_cbar["label"] = raster.attrs["units"]
    except KeyError or AttributeError:
        try:
            settings_cbar["label"] = raster.attrs["unit"]
        except KeyError or AttributeError:
            pass

    if kwargs_cbar is not None:
        settings_cbar.update(kwargs_cbar)

    # Make figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot raster
    ax1 = ax.imshow(raster, cmap=cmap, norm=norm, **settings_raster)

    # Set ax imits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Make triangles white if data is not larger/smaller than legend_levels-range
    if float(raster.max().compute()) < legend_levels[-1]:
        ax1.cmap.set_over("#FFFFFF")
    if float(raster.min().compute()) > legend_levels[0]:
        ax1.cmap.set_under("#FFFFFF")

    # Add colorbar
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    cb = fig.colorbar(ax1, cmap=cmap, norm=norm, cax=cbar_ax, **settings_cbar)

    # Add overlays
    for overlay in overlays:
        tmp = overlay.copy()
        gdf = tmp.pop("gdf")
        gdf.plot(ax=ax, **tmp)

    # Return
    return fig, ax
