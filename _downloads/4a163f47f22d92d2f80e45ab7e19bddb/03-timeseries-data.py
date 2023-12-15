"""
Time series data and Pandas
===========================

Handling time series is major part of geohydrology and groundwater modeling.
Time series data come in more than one form:

* Time series linked to a point location, for example a measured groundwater
  level at a specific location.
* Time series of a spatially continuous nature, for example model output of
  calculated head for a specific model layer.

We typically represent time series data at points as a
:py:class:`pandas.DataFrame`, despite the (apparent) match with GeoDataFrames.
The issue is that a GeoDataFrame has to store the geometry for every row: this
means many duplicated geometries. Fortunately, pandas' `group by`_
(split-apply-combine) functionality provides a (fairly) convenient way of
working with time series data of many points.

Pandas provides many tools for working with time series data, such as:

* Input and output to many tabular formats, such as CSV or Excel;
* Data selection;
* Filling or interpolating missing data;
* Resampling to specific frequencies;
* Plotting.

Timeseries at point locations
-----------------------------

iMOD represents time series at points in an IPF format. This format stores
its data as:

* A "mother" file containing the x and y coordinates of the point. Each
  point can be associated with a timeseries with a label.
* A timeries file for every point.

These files can be read via :py:func:`imod.ipf.read`. The ``read`` function
will read the mother file, and follow its labels, reading every associated
timeseries file as well. Finally, these are merged into a single large table;
the properties of the point (e.g. the x,y coordinates) are duplicated for every
row.

.. note::

    This may seem wasteful, but:

    * There are few data structures available for storing point data with
      associated time series. For example: xarray can store the point location
      as coordinates, but every point will need to share its time axis -- the
      same time window for every point and the same time resolution.
    * There are equally few file formats suitable for this data. A single large
      table is supported by many file formats.
    * Pandas `group by`_ functionality is quite fast.

.. _group by: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
.. _narrow: https://en.wikipedia.org/wiki/Wide_and_narrow_data

"""
