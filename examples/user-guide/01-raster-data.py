"""
Raster data and xarray
======================

Geospatial data primarily comes in two forms: raster data and vector data.
This guide focuses on the first.

Raster data consists of rows and columns of rectangular cells. Their location
in space is defined by the number of rows, the number of columns, a cell size
along the rows, a cell size along the columns, the origin (x, y), and
optionally rotation (x, y) -- an `affine`_ matrix.

Typical examples of file formats containing raster dat are:

* GeoTIFF
* ESRII ASCII
* netCDF
* idf (iMOD 5 format)

In groundwater modeling, data commonly stored in raster format are:

* Layer topology: the tops and bottoms of geohydrological layers
* Layer properties: conductivity of aquifers and aquitards
* Model output: heads or cell by cell flows

These data consist of values for every single cell.

"""

# _affine: https://www.perrygeo.com/python-affine-transforms.html

print("hello world")
