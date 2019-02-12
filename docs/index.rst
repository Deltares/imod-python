.. imod documentation master file, created by
   sphinx-quickstart on Tue Apr 10 12:38:06 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

iMOD-Python documentation
=========================

Work with `iMOD <http://oss.deltares.nl/web/imod>`__ MODFLOW models in
Python.

Documentation: https://deltares.gitlab.io/imod-python/

Source code: https://gitlab.com/deltares/imod-python

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   imod

Getting started
===============

.. code:: python

   import imod

   # read and write IPF files to pandas DataFrame
   df = imod.ipf.load('wells.ipf')
   imod.ipf.save('wells-out.ipf', df)

   # get all calculated heads in a xarray DataArray
   # with dimensions time, layer, y, x
   da = imod.idf.load('path/to/results/head_*.idf')

Introduction
============

The imod Python package is an addition to iMOD and iMODFLOW, intended to facilitate
working with groundwater models from Python. It does this by supporting reading and
writing of the different iMOD file formats to existing objects often used in Python
data processing.

IDF - iMOD Data Format
----------------------
IDF is the binary raster format of iMOD. One file contains a X and Y 2 dimensional grid.
Using a set of file name conventions more dimensions such as ``time`` and ``layer`` are
added, for example: ``head_20181113_l3.idf`` for layer 3 and timestamp ``2018-11-13``.
This package maps IDF files to and from the N dimensional labeled arrays of
`xarray.DataArray <http://xarray.pydata.org/en/stable/data-structures.html#dataarray>`__,
using :meth:`imod.idf.load` and :meth:`imod.idf.save`, or, to read multiple parameters
at the same time, :meth:`imod.idf.loadset`.

For more information on how to work with ``xarray.DataArray`` objects, we refer to the
xarray documentation. Note that converting GIS raster formats to IDF is supported
through `xarray.open_rasterio <http://xarray.pydata.org/en/stable/generated/xarray.open_rasterio.html#xarray.open_rasterio>`__,
followed by :meth:`imod.idf.save`.

IPF - iMOD Point File
---------------------
IPF files are text files used for storing tabular point data such as timeseries and
borehole measurements. In the imod Python package these files are read in as
`pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/>`__. Pandas is a
popular package that makes analysis and processing of tabular data easy, and provides
many input and output options, which in turn enables us to convert for instance
existing CSV or Excel files to IPF files. The primary functions for reading and writing
IPF files are :meth:`imod.ipf.load` and :meth:`imod.ipf.save`.

Authors
=======
This Python package was written primarily by Martijn Visser and Huite Bootsma at Deltares.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
