Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

[Unreleased]
------------

[0.9.0] - 2020-01-19
------------

Added
~~~~~
-  IDF files representing data of arbitrary dimensionality can be opened and
   saved. This enables reading and writing files with more dimensions than just x,
   y, layer, and time.
-  Added multi-species support for (:mod:`imod.wq`)
-  GDAL rasters representing N-dimensional data can be opened and saved similar to (:mod:`imod.idf`) in (:mod:`imod.rasterio`)
-  Writing GDAL rasters using (:meth:`imod.rasterio.save`) and (:meth:`imod.rasterio.write`) auto-detects GDAL driver based on file extension
-  64-bit IDF files can be opened (:meth:`imod.idf.open`)
-  64-bit IDF files can be written using (:meth:`imod.idf.save`) and (:meth:`imod.idf.write`) using keyword ``dtype=np.float64``
-  ``sel`` and ``isel`` methods to ``SeawatModel`` to support taking out a subdomain
-  Docstrings for the Modflow 6 classes in :mod:`imod.mf6`
-  :meth:`imod.select.upper_active_layer` function to get the upper active layer from ibound ``xr.DataArray``

Changed
~~~~~~~

-  :func:`imod.idf.read` is deprecated, use :mod:`imod.idf.open` instead
-  :func:`imod.rasterio.read` is deprecated, use :mod:`imod.rasterio.open` instead

Fixed
~~~~~

-  (:meth:`imod.prepare.reproject`) working instead of silently failing when given a ``"+init=ESPG:XXXX`` CRS string

[0.8.0] - 2019-10-14
--------------------

Added
~~~~~
-  Laplace grid interpolation (:meth:`imod.prepare.laplace_interpolate`)
-  Experimental Modflow 6 structured model write support (:mod:`imod.mf6`)
-  More supported visualizations (:mod:`imod.visualize`)
-  More extensive reading and writing of GDAL raster in (:mod:`imod.rasterio`)

Changed
~~~~~~~

-  The documentation moved to a custom domain name: https://imod.xyz/

[0.7.1] - 2019-08-07
--------------------

Added
~~~~~
-  ``"multilinear"`` has been added as a regridding option to ``imod.prepare.Regridder`` to do linear interpolation up to three dimensions.
-  Boundary condition packages in ``imod.wq`` support a method called ``add_timemap`` to do cyclical boundary conditions, such as summer and winter stages.

Fixed
~~~~~

-  ``imod.idf.save`` no longer fails on a single IDF when it is a voxel IDF (when it has top and bottom data).
-  ``imod.prepare.celltable`` now succesfully does parallel chunkwise operations, rather than raising an error.
-  ``imod.Regridder``'s ``regrid`` method now succesfully returns ``source`` if all dimensions already have the right cell sizes, rather than raising an error.
-  ``imod.idf.open_subdomains`` is much faster now at merging different subdomain IDFs of a parallel modflow simulation.
-  ``imod.idf.save`` no longer suffers from extremely slow execution when the DataArray to save is chunked (it got extremely slow in some cases).
-  Package checks in ``imod.wq.SeawatModel`` succesfully reduces over dimensions.
-  Fix last case in ``imod.prepare.reproject`` where it did not allocate a new array yet, but returned ``like`` instead of the reprojected result.

[0.7.0] - 2019-07-23
--------------------

Added
~~~~~

-  :mod:`imod.wq` module to create iMODFLOW Water Quality models
-  conda-forge recipe to install imod (https://github.com/conda-forge/imod-feedstock/)
-  significantly extended documentation and examples
-  :mod:`imod.prepare` module with many data mangling functions
-  :mod:`imod.select` module for extracting data along cross sections or at points
-  :mod:`imod.visualize` module added to visualize results
-  :func:`imod.idf.open_subdomains` function to open and merge the IDF results of a parallelized run
-  :func:`imod.ipf.read` now infers delimeters for the headers and the body
-  :func:`imod.ipf.read` can now deal with heterogeneous delimiters between multiple IPF files, and between the headers and body in a single file

Changed
~~~~~~~

-  Namespaces: lift many functions one level, such that you can use e.g. the function ``imod.prepare.reproject`` instead of ``imod.prepare.reproject.reproject``

Removed
~~~~~~~

-  All that was deprecated in v0.6.0

Deprecated
~~~~~~~~~~

-  :func:`imod.seawat_write` is deprecated, use the write method of :class:`imod.wq.SeawatModel` instead
-  :func:`imod.run.seawat_get_runfile` is deprecated, use :mod:`imod.wq` instead
-  :func:`imod.run.seawat_write_runfile` is deprecated, use :mod:`imod.wq` instead

[0.6.1] - 2019-04-17
--------------------

Added
~~~~~

-  Support nonequidistant models in runfile

Fixed
~~~~~

-  Time conversion in runfile now also accepts cftime objects

[0.6.0] - 2019-03-15
--------------------

The primary change is that a number of functions have been renamed to
better communicate what they do.

The ``load`` function name was not appropriate for IDFs, since the IDFs
are not loaded into memory. Rather, they are opened and the headers are
read; the data is only loaded when needed, in accordance with
``xarray``'s design; compare for example ``xarray.open_dataset``. The
function has been renamed to ``open``.

Similarly, ``load`` for IPFs has been deprecated. ``imod.ipf.read`` now
reads both single and multiple IPF files into a single
``pandas.DataFrame``.

Removed
~~~~~~~

-  ``imod.idf.setnodataheader``

Deprecated
~~~~~~~~~~

-  Opening IDFs with ``imod.idf.load``, use ``imod.idf.open`` instead
-  Opening a set of IDFs with ``imod.idf.loadset``, use
   ``imod.idf.open_dataset`` instead
-  Reading IPFs with ``imod.ipf.load``, use ``imod.ipf.read``
-  Reading IDF data into a dask array with ``imod.idf.dask``, use
   ``imod.idf._dask`` instead
-  Reading an iMOD-seawat .tec file, use ``imod.tec.read`` instead.

Changed
~~~~~~~

-  Use ``np.datetime64`` when dates are within time bounds, use
   ``cftime.DatetimeProlepticGregorian`` when they are not (matches
   ``xarray`` defaults)
-  ``assert`` is no longer used to catch faulty input arguments,
   appropriate exceptions are raised instead

Fixed
~~~~~

-  ``idf.open``: sorts both paths and headers consistently so data does
   not end up mixed up in the DataArray
-  ``idf.open``: Return an ``xarray.CFTimeIndex`` rather than an array
   of ``cftime.DatimeProlepticGregorian`` objects
-  ``idf.save`` properly forwards ``nodata`` argument to ``write``
-  ``idf.write`` coerces coordinates to floats before writing
-  ``ipf.read``: Significant performance increase for reading IPF
   timeseries by specifying the datetime format
-  ``ipf.write`` no longer writes ``,,`` for missing data (which iMOD
   does not accept)

[0.5.0] - 2019-02-26
--------------------

Removed
~~~~~~~

-  Reading IDFs with the ``chunks`` option

Deprecated
~~~~~~~~~~

-  Reading IDFs with the ``memmap`` option
-  ``imod.idf.dataarray``, use ``imod.idf.load`` instead

Changed
~~~~~~~

-  Reading IDFs gives delayed objects, which are only read on demand by
   dask
-  IDF: instead of ``res`` and ``transform`` attributes, use ``dx`` and
   ``dy`` coordinates (0D or 1D)
-  Use ``cftime.DatetimeProlepticGregorian`` to support time instead of
   ``np.datetime64``, allowing longer timespans
-  Repository moved from ``https://gitlab.com/deltares/`` to
   ``https://gitlab.com/deltares/imod/``

Added
~~~~~

-  Notebook in ``examples`` folder for synthetic model example
-  Support for nonequidistant IDF files, by adding ``dx`` and ``dy``
   coordinates

Fixed
~~~~~

-  IPF support implicit ``itype``

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
