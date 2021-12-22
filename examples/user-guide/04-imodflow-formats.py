"""
Overview
========

Introduction
------------

The imod Python package can be used as addition to iMOD 5 with iMODFLOW, intended to facilitate
working with groundwater models from Python. It does this by supporting reading and
writing of the different iMOD file formats to existing objects often used in Python
data processing.

IDF - iMOD  Data Format
----------------------
IDF is the binary raster format of iMOD 5. One file contains a X and Y 2
dimensional grid. Using a set of file name conventions more dimensions such as
``time`` and ``layer`` are added, for example: ``head_20181113_l3.idf`` for
layer 3 and timestamp ``2018-11-13``.  This package maps IDF files to and from
the N dimensional labeled arrays of `xarray.DataArray`_ using
:meth:`imod.idf.open` and :meth:`imod.idf.save`, or, to read multiple
parameters at the same time, :meth:`imod.idf.open_dataset`.

For more information on how to work with ``xarray.DataArray`` objects, we refer
to the xarray documentation. Note that converting GIS raster formats to IDF is
supported through `xarray.open_rasterio`_ followed by :meth:`imod.idf.save`.

IPF - iMOD Point File
---------------------
IPF files are text files used for storing tabular point data such as timeseries
and borehole measurements. In the imod Python package these files are read in
as `pandas.DataFrame`_. Pandas is a popular package that makes analysis and
processing of tabular data easy, and provides many input and output options,
which in turn enables us to convert for instance existing CSV or Excel files to
IPF files. The primary functions for reading and writing IPF files are
:meth:`imod.ipf.read` and :meth:`imod.ipf.save`.

PRJ - iMOD Project File
-----------------------
A PRJ file describes the configuration of a iMOD 5 model, it is a list of files that 
are associated to model layers and/or time steps. From a PRJ aMF6,  MF2005 or Runfile 
can be configured for a specified number of model layers and/or transient periods.

Other iMOD 5 file formates
--------------------------
Other iMOD 5 file formats like GEN (for vector and polygon data), ISG (vector surface 
water data) and IFF (iMOD Flowpath File) are not supported within the imod Python package. 


"""
