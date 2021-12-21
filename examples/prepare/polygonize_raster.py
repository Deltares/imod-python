"""
Polygonize raster
=================

iMOD Python also provides convenience functions to polygonize rasters.
"""

# %%
import imod
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = -1

# %%
# We'll start off by creating an example raster ``lake_grid`` to convert to
# polygons. This is similar to the `Rasterize shapefiles` example.

temp_dir = imod.util.temporary_directory()
lakes = imod.data.lakes_shp(temp_dir)

# Create dummy grid
xmin = 90950.0
xmax = 115650.0
dx = 200

ymin = 445850.0
ymax = 467550.0
dy = -200.0

like_2d = imod.util.empty_2d(dx, xmin, xmax, dy, ymin, ymax)

# Rasterrize the shapes
lake_grid = imod.prepare.rasterize(lakes, like=like_2d)

# %%
# Our raster looks like this:
fig, ax = plt.subplots()
lake_grid.plot(ax=ax)

# %%
# Polygonize the lakes
polygonized_lakes = imod.prepare.polygonize(lake_grid)

polygonized_lakes.head(5)

# %%
# This also polygonized the areas with np.nan. So we can drop those, using
# regular pandas functionality

polygonized_lakes = polygonized_lakes.dropna()

polygonized_lakes.head(5)

# %%
# Plotted, we see a similar picture to the plotted raster
fig, ax = plt.subplots()
polygonized_lakes.plot(ax=ax)

# %%
# Since it is a GeoDataFrame, we can now do vector operations. Like computing
# the centroids and plotting them as points.

centroids = polygonized_lakes.centroid

fig, ax = plt.subplots()
polygonized_lakes.plot(ax=ax)
centroids.plot(ax=ax, color="k")
