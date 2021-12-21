"""
Plot maps
=============

The ``imod.visualize.plot_map`` functionality of iMOD Python allows to create
customized plots.

"""

# %%
# Import the necessary packages:

import imod
import numpy as np

# %%
# Import the input data to plot:

tempdir = imod.util.temporary_directory()

lakes = imod.data.lakes_shp(tempdir)
surface_level = imod.data.ahn()["ahn"]

# %%
# It is necessary to define the Matplotlib colorbar to be used and the levels
# for the legend as a list.
colors = "RdYlBu_r"
levels = np.arange(-15, 17.5, 2.5)

# %%
# The next lines show the simplest way to plot the raster.
imod.visualize.plot_map(surface_level, colors, levels)

# %%
# It is also possible to add an overlay to the previous map

overlays = [{"gdf": lakes, "facecolor": "black", "alpha": 0.3}]

imod.visualize.plot_map(surface_level, colors, levels, overlays=overlays)

# %%
# Label the colorbar as follows:
imod.visualize.plot_map(
    surface_level, colors, levels, kwargs_colorbar={"label": "Surface level (m)"}
)

# %%
# And to include a basemap:
import contextily as ctx

src = ctx.providers.Stamen.TonerLite
imod.visualize.plot_map(
    surface_level
    colors,
    levels,
    basemap=src,
    kwargs_basemap={"alpha": 0.6},
    overlays=overlays,
    kwargs_colorbar={"label": "Surface level (m)"},
)
