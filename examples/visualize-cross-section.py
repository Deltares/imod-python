"""
Visualize a cross seciton
=========================

Visualization Tools in iMOD python to visualize cross section
"""
import re

import imod

###############################################################################
# Read the input
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We are going to read the top and bot data from Brabant_steady-state example directory.
# We have 19 layers model, 601 columns, and 450 rows.
# The cell is a rectangular cell with width of 250.

top = imod.rasterio.open(
    "data/brabant/topbot/rl*.tif",
    pattern=re.compile(r"(?P<name>[\w]+)L(?P<layer>[\d+]*)", re.IGNORECASE),
)
bot = imod.rasterio.open(
    "data/brabant/topbot/TH*.tif",
    pattern=re.compile(r"(?P<name>[\w]+)H(?P<layer>[\d+]*)", re.IGNORECASE),
)
nodata = -9999.0
top = top.where(top > -9990.0)
bot = bot.where(bot > -9990.0)

###############################################################################
# Plotting cross section with imod cross section visualization function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This function can be used to  to draw cross-sections and draw cell boundaries accurately.
# If aquitard presents, it can be plotted on top of the cross-section, by providing a DataArray with the aquitard location.
# ``top`` and ``bottom`` coordinates are required on the DataArray.

top = top.assign_coords(top=(("layer", "y", "x"), top))
top = top.assign_coords(bottom=(("layer", "y", "x"), bot))
cross = imod.visualize.cross_section(
    top.isel(y=200),
    colors="RdYlBu_r",
    levels=range(-400, 200, 100),
    kwargs_colorbar={"label": "layer elevation at y = 384875"},
)
