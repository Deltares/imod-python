"""
Unstructured Grids
==================

MODFLOW 6 supports unstructured grids. Unlike raster data, the connectivity of
unstructured grids is irregular. This means that the number of neighboring
cells is not constant; for structured grids, every cell has 4 neighbors (in
2D), or 6 neighbors (in 3D).

The package we use to handle unstructured grids is called `xugrid
<https://github.com/Deltares/xugrid>`_. It is a Python package that provides a
data structure for unstructured grids, and allows for plotting and analysis of
these grids. It is built on top of `xarray
<https://xarray.pydata.org/en/stable/>`_`, so handling data is similar to how
you would handle raster data in xarray, with some differences.
"""

# %% 
# 
# Let's first load some sample data from xugrid.data module. The example data
# is a triangular grid of the elevation of the Netherlands.

import xugrid

uda = xugrid.data.elevation_nl()

uda

# %%
#
# Note that this data is stored differently than a raster dataset. We don't see
# an x coordinate and a y coordinate here. So where is this spatial information?
# It's in this ``grid`` attribute, which is accesses via the ``ugrid`` accessor.
# This is a special accessor that provides access to the unstructured grid
# properties of the dataset.

grid = uda.ugrid.grid
grid

# %%
# 
# The grid has a number of properties, such as the number of cells, the
# coordinates of the vertices, and the connectivity of the cells. The latter is
# very important when dealing with unstructured grids, as it defines how the
# cells are connected to each other. This is different from structured grids,
# where you don't need to specify this as there it follows from the rows and
# columns. Let's take a look at the connectivity of the grid.

grid.format_connectivity_as_dense(grid.face_face_connectivity)

# %%
#
# This shows the connectivity of the cells in the grid. Each row represents a
# cell, and the columns represent the neighboring cells. Because this is a grid
# with triangles, we can see three columns in the connectivity matrix. -1
# indicates that a cell is not connected to a neighbour. For example, the first
# cell is only connected to one cell.
#
# You can imagine that plotting unstructured grids is quite a hassle, as you
# need to provide quite some information to the plotting function. Fortunately,
# the xugrid package provides a convenient way to plot unstructured grids. Let's
# plot the grid:

grid.plot()

# %%
#
# We can plot the grid with the data values as well:

uda.ugrid.plot()

# %%
#
# Note that we need to call the ``ugrid`` accessor to plot the data. This is
# because the data is stored in a different format than raster data, and we need
# to tell that xugrid needs to be used for plotting, and not the regular xarray
# plotting methods.


# %%
