Data In/Out
-----------

Import IDF file
~~~~~~~~~~~~~~~

With :func:`imod.idf.open`:

.. code-block:: python

    da = imod.idf.open("bottom_l1.idf")


Import multiple IDF files
~~~~~~~~~~~~~~~~~~~~~~~~~

With :func:`imod.idf.open`:

.. code-block:: python

    da = imod.idf.open("bottom_l*.idf")
    

Import IPF file
~~~~~~~~~~~~~~~

With :func:`imod.ipf.read`:

.. code-block:: python

    df = imod.ipf.read("timeseries.ipf")
    

Import netCDF file as Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ds = xr.open_dataset("dataset.nc")
    
Import a single netCDF variable as DataArray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    da = xr.open_dataarray("variable.nc")
    

Convert structured data to UGRID netCDF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With :func:`imod.util.to_ugrid2d`:

.. code-block:: python

    ugrid_ds = imod.util.to_ugrid2d(da)
    ugrid_ds.to_netcdf("ds_ugrid.nc")
 