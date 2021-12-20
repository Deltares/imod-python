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

In Python, such data can be represented by a ``geopandas.GeoDataFrame``.
Essentially, geopandas is a pandas DataFrame to store tabular data (the
attribute table), and adds a geometry column to store the geospatial
coordinates.

Geopandas can read variety of file formats:
"""

# import geopandas as gpd
# import pandas as pd
#
# gdf_from_shp = gpd.read_file("../examples/data/2017-Waterschappen.shp")
# gdf_from_shp.head()  # show the first five rows

##############################################################################
# This geodataframe contains all the data from the shapefile. Note the geometry
# column.
#
# A GeoDataFrame of points can also be easily generated from a tabular data.
# We'll use pandas to read a CSV file, and convert the x and y columns to point
# geometry.


# df = pd.read_csv("../examples/data/point-data.csv")
# gdf_from_csv = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df["x"], y=df["y"]))
