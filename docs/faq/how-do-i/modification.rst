Data modification
-----------------

If-then-else
~~~~~~~~~~~~

Remove all values over 5.0:

.. code-block:: python

    da = da.where(da < 5.0)
    
Let's say we want to replace all values ``< 5`` by a ``1``, and all values ``>=
5`` by a ``2``. The easiest way to do this is by using :py:func:`imod.util.where`:

.. code-block:: python

    condition = old < 5.0
    new = imod.util.where(condition, if_true=1, if_false=2)

This can also be done with xarray directly, but is less convenient:

.. code-block:: python

    condition = old < 5.0
    new = xr.full_like(old, 2.0)
    new = new.where(condition, other=1)
 
Alternatively:

.. code-block:: python

    condition = old < 5
    new = xr.where(condition, x=1.0, y=2.0)

.. note::

    When ``condition`` does not have the same dimension as ``x`` or ``y``, you
    may end up with an unexpected dimension order; ``da = da.where(...)``
    always preserves the dimension order of ``da``. Using
    :py:func:`imod.util.where` avoids this.

.. note::

    Xarray uses NaN (Not-A-Number) for nodata / fill values. NaN values have
    special properties in inequality operations: ``(np.nan < 5) is False``, but also
    ``(np.nan >= 5) is False`` as well.
    
    For this reason, :py:func:`imod.util.where` defaults to keeping NaNs by default:
    ``imod.util.where(old < 5.0, old, 5.0)`` will preserve the NaN values, while
    ``xr.where(old < 5.0, old, 5.0)`` will fill the NaN values with 5.0

Conditional evaluation
~~~~~~~~~~~~~~~~~~~~~~

Let's say we want to select values between 5 and 10.

Avoid:

.. code-block:: python

    condition1 = da > 5
    condition2 = da < 10
    condition = condition1 and condition2
    
Do instead:

.. code-block:: python

    condition1 = da > 5
    condition2 = da < 10
    condition = condition1 & condition2

The reason is that ``and`` does work on the individual values, but expects
something like a boolean (``True`` or ``False``). To do element-wise
conditional evaluation on the individual values, use:

* and: ``&``
* or: ``|``
* not: ``~``
* xor: ``^``
  
To check there are no NaNs anywhere, use "reductions" such as ``.all()`` or
``.any()`` to reduce the array to a single boolean:

.. code-block:: python

    has_nodata = da.notnull().any()

Arithmetic
~~~~~~~~~~

.. code-block:: python

    da3 = da1 * da2 + 5.0
    
Make sure the grids have the same spatial coordinates.

Change cellsize (and extent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nearest neighbor:

.. code-block:: python

    regridder = imod.prepare.Regridder(source, destination, method="nearest")
    out = regridder.regrid(source)
    
Area weighted mean:

.. code-block:: python

    regridder = imod.prepare.Regridder(source, destination, method="mean")
    out = regridder.regrid(source)
    
Change time resolution
~~~~~~~~~~~~~~~~~~~~~~

From e.g. hourly data to daily average:

.. code-block:: python

    new = da.resample(time="1D").mean()
    
See `xarray documentation on resampling`_.


Select along a single layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sel()`` is "key" selection, this selects the layer named "1":

.. code-block:: python

    da_layer1 = da.sel(layer=1)

``isel()`` is "index" selection, this selects the first layer:

.. code-block:: python

    da_firstlayer = da.isel(layer=0)
    
Select part of the data
~~~~~~~~~~~~~~~~~~~~~~~

Generally, raster data is y-descending, so ``ymax`` comes before ``ymin``:

.. code-block:: python

    da_selection = da.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
  
Create an empty raster
~~~~~~~~~~~~~~~~~~~~~~

For just a two dimensional x-y grid:

.. code-block:: python

    da = imod.util.empty_2d(dx, xmin, xmax, dy, ymin, ymax)
    
For a three dimensional version:
    
.. code-block:: python

    da = imod.util.empty_3d(dx, xmin, xmax, dy, ymin, ymax, layer=[1, 2, 3])

For a time varying 2d grid:

.. code-block:: python

    da = imod.util.empty_2d_transient(
        dx, xmin, xmax, dy, ymin, ymax, time=pd.date_range("2020-01-01", "2020-02-01")
    )

For a time varying 3d grid:

.. code-block:: python

    da = imod.util.empty_3d_transient(
        dx,
        xmin,
        xmax,
        dy,
        ymin,
        ymax,
        layer=[1, 2, 3],
        time=pd.date_range("2020-01-01", "2020-02-01")
    )

Fill/Interpolate nodata
~~~~~~~~~~~~~~~~~~~~~~~

To do nearest neighbor interpolation:

.. code-block:: python

    new = imod.prepare.fill(da_with_gaps)
    
To do interpolation along a single dimension:

.. code-block:: python

    new = da_with_gaps.interpolate_na(dim="x") 
    
See the `xarray documentation on interpolation of NaN values`_.
    
To do interpolation in time, see `Change time resolution`_.
    
To do Laplace interplation (using a linear equation, similar to a groundwater
model with constant conductivity):

.. code-block:: python

    da = imod.prepare.laplace_interpolate(with_gaps)
    
Rasterize polygon data
~~~~~~~~~~~~~~~~~~~~~~

A geopandas GeoDataFrame can be rasterized by providing a sample DataArray for
``like`` in :py:func:`imod.prepare.rasterize`:

.. code-block:: python

   rasterized = imod.prepare.rasterize(
       geodataframe,
       column="column_name_in_geodataframe",
       like=like,
    ) 
   
For large vector datasets, reading the files into a geodataframe can take
longer dan the rasterization step. To avoid this, it's possible to skip loading
the data altogether with :py:func:`imod.prepare.gdal_rasterize`

.. code-block:: python

   rasterized = imod.prepare.gdal_rasterize(
       path="path-to-vector-data.shp",
       column="column_name_in_vector_data",
       like=like,
    ) 

Smooth data
~~~~~~~~~~~

We can use a `convolution`_ to smooth:

.. code-block:: python

    kernel = np.ones((1, 10, 10))
    kernel /= kernel.sum()
    da.values = scipy.ndimage.convolve(da.values, kernel)

Zonal statistics
~~~~~~~~~~~~~~~~

To compute a mean:

.. code-block:: python

    mean = da.groupby(zones).mean("stacked_y_x")

To compute a sum:

.. code-block:: python

    sum = da.groupby(zones).sum("stacked_y_x")
    
.. note:: 

    This is not the most efficient way of computing zonal statistics. If it
    takes a long time or consumes a lot of memory, see e.g. `xarray-spatial's
    zonal stats`_ function.

Force loading into memory / dask array to numpy array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    da = da.compute()
    
Alternatively:

.. code-block:: python

    da = da.load()
    
Select a single variable from a dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select ``"kd"`` from dataset ``ds``:

.. code-block:: python

    da_kd = ds["kd"]
    
Select points (from a vector dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    geometry = geodataframe.geometry
    x = geometry.x
    y = geometry.y
    selection = imod.select.points_values(da, x=x, y=y)

For time series analysis, converting into a pandas DataFrame may be useful:

.. code-block:: python

    df = selection.to_dataframe()

Sum properties over layers
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    total_kd = da_kd.sum("layer")

.. _xarray documentation on resampling: https://xarray.pydata.org/en/stable/user-guide/time-series.html#resampling-and-grouped-operations.
.. _xarray documentation on interpolation of NaN values: https://xarray.pydata.org/en/stable/generated/xarray.DataArray.interpolate_na.html
.. _convolution: https://en.wikipedia.org/wiki/Convolution
.. _xarray-spatial's zonal stats: https://xarray-spatial.org/reference/_autosummary/xrspatial.zonal.stats.html