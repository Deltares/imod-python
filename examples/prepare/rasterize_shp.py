"""
Rasterize shapefiles
======================

Importing the necessary packages:
"""

import imod
import matplotlib.pyplot as plt

# %%
# Get the example shapes
temp_dir = imod.util.temporary_directory()
lakes = imod.data.lakes_shp(temp_dir)

# %%
# We'll need a dummy grid which we will use as a reference for for rasterizing
# the shapefile. We are just going to create one with the convenience function.

xmin = 90950.0
xmax = 115650.0
dx = 100

ymin = 445850.0
ymax = 467550.0
dy = -100.0

like_2d = imod.util.empty_2d(dx, xmin, xmax, dy, ymin, ymax)

# %%
# Rasterrize the shapes
lake_grid = imod.prepare.rasterize(lakes, like=like_2d)

# Plot
fig, ax = plt.subplots()
lake_grid.plot.imshow(ax=ax)

# %%
# To rasterize on a different grid, create a dummy grid

dx_coarse = 200
dy_coarse = -200
like_2d_coarse = imod.util.empty_2d(dx_coarse, xmin, xmax, dy_coarse, ymin, ymax)

lake_grid_coarse = imod.prepare.rasterize(lakes, like=like_2d_coarse)

# Plot
fig, ax = plt.subplots()
lake_grid_coarse.plot.imshow(ax=ax)
