User Guide
==========

.. toctree::
   :maxdepth: 1
   :hidden:

The imod Python package builds on top of popular Python packages like
`xarray`_, `pandas`_, and `geopandas`_ to prepare and analyze MODFLOW models.
This user guide will introduce these packages and their data structures, and
explains how they relate to the many forms of data we come across while
modeling groundwater flow.

This user guide is not an exhaustive explanation of these packages and their
data structures. Rather this guide intends to introduce the packages, explain
their roles, and how they fit together to help with groundwater modeling.

The imod package provides a link between these packages and groundwater
modeling.Pandas, geopandas, and xarray already provide a great deal of
capabilities and features; imod expands these capabilities when they are
(i)MODFLOW specific or when existing capabilities are too limited or too slow.

Specifically:

* input and output to (i)MODFLOW specific file formats.
* data preparation, mostly "GIS-like" routines: raster to vector and vice versa,
  changing cell sizes, data selection, gapfilling, etc.
* overview statistics, water balance, etc,
* visualization of groundwater heads, cross sections, 3D animations, etc.

Nearly every function in imod consumes and produces xarray, pandas, or
geopandas data structures. Therefore, this guide first introduces these data
structures. Secondly, it will demonstrate how a modeling workflow is set up. 

.. toctree::
   :titlesonly:
   :hidden:
   
   examples/index
 
.. _xarray: http://xarray.pydata.org
.. _pandas: http://pandas.pydata.org
.. _geopandas: http://geopandas.org
