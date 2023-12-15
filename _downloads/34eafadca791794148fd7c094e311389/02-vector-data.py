"""
Vector data and Geopandas
=========================

Geospatial data primarily comes in two forms: raster data and vector data.
This guide focuses on the latter.

Typical examples of file formats containing vector data are:

* ESRI shapefile
* GeoJSON
* Geopackage

Vector data consist of vertices (corner points), optionally connected by paths.
The three primary categories of vector data are:

* Points
* Lines
* Polygons

In groundwater modeling, typical examples of each are:

* Pumping wells, observation wells, boreholes
* Canals, ditches, waterways
* Lakes, administrative boundaries, land use

These data consist of geospatial coordinates, indicating the location in space
and a number of attributes: for a canal, this could be parameters like its
width, depth, and water level. In GIS software like QGIS, the geometry is
visible in the map view, and the attributes can inspected via e.g. the
attribute table.

In Python, such data can be represented by a
:py:class:`geopandas.GeoDataFrame`. Essentially, geopandas is a pandas
DataFrame to store tabular data (the attribute table), and adds a geometry
column to store the geospatial coordinates.
"""
# %%
import geopandas as gpd
import numpy as np

import imod

tempdir = imod.util.temporary_directory()
gdf = imod.data.lakes_shp(tempdir / "lake")
gdf.iloc[:5, -3:]  # first 5 rows, last 3 columns

# %%
# This geodataframe contains all the data from the shapefile. Note the geometry
# column. The geometry can be plotted:

gdf.plot()

# %%
# A GeoDataFrame of points can also be easily generated from pairs of x and y
# coordinates.

x = np.arange(90_000.0, 120_000.0, 1000.0)
y = np.arange(450_000.0, 480_000.0, 1000.0)

geometry = gpd.points_from_xy(x, y)
points_gdf = gpd.GeoDataFrame(geometry=geometry)

points_gdf.plot()

# %%
# An important feature of every geometry is its geometry type:

gdf.geom_type

# %%
# As expected, the points are of the type ... Point:

points_gdf.geom_type

# %%
# Input and output
# ----------------
#
# Geopandas supports many vector file formats. It wraps `fiona`_, which in turns
# wraps `OGR`_, which is a part of `GDAL`_. For example, the lake polygons above
# are loaded from an ESRI Shapefile:

filenames = [path.name for path in (tempdir / "lake").iterdir()]
print("\n".join(filenames))

# %%
# They can be easily stored into more modern formats as well, such as
# `GeoPackage`_:

points_gdf.to_file(tempdir / "points.gpkg")
filenames = [path.name for path in tempdir.iterdir()]
print("\n".join(filenames))

# %%
# ... and back:

back = gpd.read_file(tempdir / "points.gpkg")
back

# %%
#
# Conversion to raster
# --------------------
#
# From the perspective of MODFLOW groundwater modeling, we are often interested
# in the properties of cells in specific polygons or zones. Refer to the
# examples or the API reference for ``imod.prepare``.
#
# GeoPandas provides a full suite of vector based GIS operations, such as
# intersections, spatial joins, or plotting.
#
# .. _fiona: https://fiona.readthedocs.io/en/latest/manual.html
# .. _OGR: https://gdal.org/faq.html#what-is-this-ogr-stuff
# .. _GDAL: https://gdal.org/
# .. _GeoPackage: https://www.geopackage.org/
# .. _GeoPandas User Guide: https://geopandas.org/en/stable/docs/user_guide.html
