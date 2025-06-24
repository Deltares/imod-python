"""
Reading and writing with iMOD5 files in Python
==============================================

This example demonstrates how to work with iMOD5 files in Python using the
`imod` package. It shows how to save, read, and manipulate iMOD5 files,
including handling temporary directories and glob patterns for file selection.

For a full overview of the supported iMOD5 features in iMOD Python, see
:doc:`../faq/imod5_backwards_compatibility`.

"""

# %%
#
# Raster data (IDF)
# -----------------
#
# Raster data in iMOD5 is stored in IDF files, which is a binary format for
# raster data. The `imod.idf` module provides functions to read and write IDF
# files. One IDF can only store data in two dimensions: y and x. It is similar
# to GeoTIFF in this regard.

# We'll start off with some example data to work with. Let's load a layer model
# with geological layers.

import imod

layermodel = imod.data.hondsrug_layermodel()
layermodel

# %%
# This dataset contains multiple variables. We can take a closer look at the
# the "top" variable, which represents the top of every layer.

top = layermodel["top"]
top

# %%
#
# Note that this DataArray has three dimensions: layer, y, x.
#
# Let's create a temporary directory to store our IDF files.
tmpdir = imod.util.temporary_directory()
idf_dir = tmpdir / "idf"

# %%
#
# As mentioned before, one IDF can only store data in two dimensions: y and x.
# iMOD Python therefore will save a 3D dataset to multiple IDF files. The layer
# index is stored in the file name, as the ``_L*`` suffix. iMOD Python will take
# the last part in the provided path as the name. We therefore end our path with
# ``top``.

imod.idf.save(idf_dir / "top", top)

# %%
# The IDF files are now stored in the temporary directory. We can list them
# using the `glob` method, which allows us to use wildcards to match file names

idf_files = list(idf_dir.glob("*"))
idf_files

# %%
#
# We can also read the IDF files back into a DataArray, we can either do that by
# directly providing a list of filenames (``idf_files``) to open, or by
# providing a path with a wildcard pattern (``idf_dir / "top*"``). The latter is
# more convenient, as it will automatically match all files that start with
# ``"top"`` in ``idf_dir``. This means we can easily open the data without
# having to specify each file individually.

reloaded_top = imod.idf.open(idf_dir / "top*")
reloaded_top

# %%
#
# Note that the representation of the DataArray showed differs from the previous
# representation. This is because the DataArray is lazily loaded as a dask
# array, which means that the data is not actually loaded into memory until it
# is accessed. For example when plotting or saving data. This allows for more
# efficient memory usage, especially when working with large datasets. See the
# :doc:`06-lazy-evaluation` for more information.
#
# Let's plot the top layer:

reloaded_top.sel(layer=1).plot()

# %%
#
# Point data (IPF)
# ----------------
#
# iMOD Stores point data in iMOD Point Format (IPF) files, which is a text
# format for point data. The `imod.ipf` module provides functions to read and
# write IPF files. Point data can be used to store timeseries of point
# observations, such as groundwater heads, as well as borelogs. In this example,
# we will focus on timeseries.
#
# Let's load some example point data.

heads = imod.data.head_observations()
heads

# %%
#
# We can plot a timeseries for one specific observation point, for example the
# observation point with ID "B12A1745001".

head_selected = heads.loc[heads["id"] == "B12A1745001"]
head_selected.sort_values("time").plot(x="time", y="head")

# %%
#
# We can save this point data to an IPF file. The `imod.ipf.save` function
# allows us to save the point data to a file. We can specify the path where we
# want to save the file, as well as the data to save. Make sure to specify itype
# as 1, which is the type for timeseries data. The path should end with the name
# you want to give to your IPF.

ipf_dir = tmpdir / "ipf"
imod.ipf.save(ipf_dir / "heads", heads, itype=1)

# %%
#
# The IPF files are now stored in the temporary directory. We'll list them as
# follows:

ipf_files = list(ipf_dir.glob("*"))
ipf_files

# %%
#
# Notice that all timeseries for each observation point are stored in a single
# textfile. We can read the IPF file and accompanying text files back into a
# DataFrame.

reloaded_heads = imod.ipf.read(ipf_dir / "heads.ipf")
reloaded_heads

# %%
#
# Line data (GEN)
# ---------------
#
# iMOD stores line data in GEN files, which is a binary format for line data.
# The `imod.gen` module provides functions to read and write GEN files. We'll
# create some dummy line data to store in a GEN file. iMOD5 primarily uses GEN
# to specify Horizontal Flow Barriers (HFB).

import shapely

x = [0.0, 14.0, 36.0, 50.0, 70.0]
y = [0.0, 20.0, 10.0, 30.0, 40.0]

ls = shapely.LineString(zip(x, y))
ls

# %%
#
# We now have a geometry, but there is no data associated with it yet. We can
# create a GeoDataFrame with some data to associate with the line.

import geopandas as gpd

gdf = gpd.GeoDataFrame([100.0], geometry=[ls], columns=["resistance"])

# %%
#
# We can save this line data to a GEN file with the `imod.gen.write` function.

gen_dir = tmpdir / "gen"
# We'll have to create the directory first, as it does not exist yet.
gen_dir.mkdir(exist_ok=True)

imod.gen.write(
    gen_dir / "barrier.gen",
    gdf,
)

# %%
#
# You can see that the file is saved in the specified directory.

gen_files = list(gen_dir.glob("*"))
gen_files

# %%
#
# We can read the GEN file back into a GeoDataFrame with the `imod.gen.read`

reloaded_gdf = imod.gen.read(gen_dir / "barrier.gen")
reloaded_gdf

# %%
#
# The GEN file also supports storing "3D data", which allows you to store
# vertically oriented polygons, which you can use to insert
# partially-penetrating horizontal flow barriers in iMOD5 and iMOD Python.
#
# Let's create some 3D data first. iMOD Python has a convenience function
# to create a 3D polygon conveniently.

ztop = [0.0, 0.0, -10.0, 0.0, 0.0]
zbottom = [-20.0, -20.0, -20.0, -20.0, -20.0]

zpolygon = imod.prepare.linestring_to_trapezoid_zpolygons(x, y, ztop, zbottom)

gdf_polygon = gpd.GeoDataFrame(
    [100.0, 100.0, 100.0, 100.0], geometry=zpolygon, columns=["resistance"]
)

# %%
imod.gen.write(
    gen_dir / "barrier_3d.gen",
    gdf_polygon,
)

# %%
# This breaks for some reason?

# reloaded_gdf = imod.gen.read(gen_dir / "barrier_3d.gen")
# reloaded_gdf

# %%
#
# 1D Network (ISG)
# ----------------
#
# iMOD5 stores store 1D networks in ISG files, which is a binary format for
# primarily for surface water datasets. It supports all kinds of extra features
# such as associated bathymetries and weirs.
#
# Given its complexity, ISG is not yet supported in iMOD Python. The workaround
# is to rasterize the ISGfiles to IDF files using the iMOD5 BATCH function
# ISGGRID.
# %%
#
# Legend files (LEG)
# ------------------
#
# iMOD5 uses LEG files to store legend information for plotting.

# INSERT LEGEND FILE AND EXAMPLE HERE

# %%
#
# Project files (PRJ)
# -------------------
#
# iMOD5 uses PRJ files to store project information, basically the model
# definition. `See the example for a full overview of importing a model from a
# projectfile into a MODFLOW6 model <INSERT LINK HERE>`_.
