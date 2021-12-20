General Questions
=================

How is imod-python different from FloPy?
----------------------------------------

`FloPy`_ is the USGS supported Python package to "create, run, and post-process
MODFLOW-based models.". Creating, running, and post-processing MODFLOW-based
models is largely the goal of imod-python as well. So, what are the differences?

Supported models
~~~~~~~~~~~~~~~~

FloPy nearly fully supports a list of models and software:

* MODFLOW2005
* MODFLOW-LGR
* MODFLOW-USG
* MODFLOW6
* SEAWAT
* MODPATH
* MT3D
  
imod supports (partially):

* iMODFLOW
* iMOD-WQ
* MODFLOW6

Data structures
~~~~~~~~~~~~~~~

FloPy is built on NumPy arrays, the `fundamental package`_ for scientic
computing in Python; imod is built on xarray. To quote "`Why xarray?`_":

    Xarray introduces labels in the form of dimensions, coordinates and attributes
    on top of raw NumPy-like multidimensional arrays, which allows for a more
    intuitive, more concise, and less error-prone developer experience.

imod-python started out as a package to read iMODFLOW files (IDF, IPF) into
Python as xarray and pandas data structures. The `Why xarray?`_ page provides
more background, but we can demonstrate with an example of two tasks for a
(structured) MODFLOW6 model:

    1. selecting the heads at a specified (x, y) point;
    2. computing a mean of the heads over time.

With FloPy:

.. code-block:: python

    import flopy
    
    hds = flopy.utils.binaryfile.HeadFile("GWF_1/GWF_1.hds")
    head = hds.get_alldata()
    
    simulation = flopy.mf6.MFSimulation.load(sim_ws=".")
    model = simulation.get_model("GWF_1")
    grid = model.modelgrid
    row, column = grid.intersect(50_000.0, 30_000.0)
    selection = head[:, :, row, column]

    mean_head = head.mean(axix=0)

With imod:

.. code-block:: python

    import imod
    
    head = imod.mf6.open_hds("GWF_1/GWF_1.hds", "GWF_1/GWF_1.grb")
    
    selection = head.sel(x=50_000.0, y=30_000.0, method="nearest")
    mean_head = head.mean("time")
 
Some differences to note:

* imod always requires the binary grid file -- it needs this file to provide
  dimensions and coordinates to the head data; FloPy stores this information in
  a separate modelgrid object.
* As the xarray object knows is location in space, we can directly select values
  based on coordinates.
* imod provides us greater convenience -- the convenience of xarray over numpy, of
  coordinates (labels) over integer locations.
* These conveniences extend to many facets of modeling, e.g.: resampling in
  time, plotting, input from and output to many file formats.

Dependencies
~~~~~~~~~~~~

FloPy has only two required dependencies:

.. code::

    install_requires =
        numpy >= 1.15.0
        matplotlib >= 1.4.0

imod has too many to count:

.. literalinclude:: ../../imod-environment.yml
   :language: yaml
   :caption: imod-environment.yml

Consequently, FloPy can be easily installed with just ``pip``; ``imod``
requires the ``mamba`` or ``conda`` package managers to install correctly
(especially on Windows).

 
Large data
~~~~~~~~~~

imod has been designed to deal gracefully with large datasets. Mostly, this is
thanks to xarray's :doc:`../user-guide/06-lazy-evaluation`. However, imod also
makes use of `dask`_ to deal with datasets that do not fit in memory.

Let's revisit the example above. What if the heads file of the simulation is a
100 gigabyte and we'd like to store it as a netCDF? Using FloPy, we'd have to
write a loop, appending time steps to the netCDF file one by one (using
``hds.get_data(time=...)`` instead of ``hds.get_alldata()``). Because imod uses dask,
it will automatically process the data on a chunk-by-chunk basis with
``head.to_netcdf("head.nc")``.

.. _FloPy: https://github.com/modflowpy/flopy
.. _fundamental package: https://numpy.org/doc/stable/user/whatisnumpy.html
.. _Why xarray?: https://xarray.pydata.org/en/stable/getting-started-guide/why-xarray.html
.. _dask: https://dask.org/
