Indexing
========

Xarray is fully documented in terms of its concepts and features, and we
recommend going through it at least once. However, the documentation is
sometimes considered opaque or somewhat abstract; it is a full reference, it
lists all available operations, but doesn't necessarily tell what you have to
do to combine these operations.

The documentation below attempts to provide a more specific guide, hopefully
giving you a running start ... or at the least a walking start rather than a
crawling start.

Boolean arrays and bitwise operators
------------------------------------

Python has a number of logical operators:

* ``and``
* ``or``
* ``not``
* exclusive or, or xor, via the caret symbol: ``^``

These work on a single boolean value, or a single pair of boolean values, i.e.
``True`` and ``False``.

.. code-block:: python

   not True  # False
   True and False  # False
   True or False  # True
   not (True and False)  # True
   True ^ False  # True
   True ^ True  # False
   False ^ False # False

Additionally, numpy provides so-called bitwise operators. These work on one or
more boolean values:

.. code-block:: python

   a = np.array([True, True, False])
   b = np.array([False, True, False])
   # bitwise or
   a | b  # True, True, False
   # bitwise and
   a & b  # False, True, False
   # bitwise inversion
   ~a  # False, False, True
   ~b  # True, False, True
   # bitwise xor
   a ^ b  # True, False, False

These operators can be used to manipulate entire boolean arrays at once.

Conditional selection
---------------------

Conditional selection in xarray is done with ``.where``. The syntax is:

.. code-block:: python

   array.where(condition=boolean_array, other=otherwise)

In English this translates to:
* Take the ``array`` where the ``boolean_array`` is ``True``
* Take ``otherwise`` where the ``boolean_array`` is ``False``

For example, if we're working on a digital elevation model (DEM), we might want
exclude certain values exceeding a threshold:

.. code-block:: python

   above_sea_level = dem > 0.0
   dem_above_sea_level = dem.where(above_sea_level)

Alternatively, we might want to replace these values by the threshold.
In this case, we have to take ``nan`` values properly into account. Note:

.. code-block:: python

   np.nan > 0.0 # False
   np.nan < 0.0 # False

This means the following does not give the required result:

.. code-block:: python

   corrected_dem = dem.where(above_sea_level, other=0.0)

Since the nodata values in ``dem`` will have been filled with ``0.0`` as well.
There are multiple ways around this, by taking an additional step:

.. code-block:: python

   corrected_dem = dem.where(above_sea_level, other=0.0)
   hasdata = np.isfinite(dem)
   corrected_dem = corrected_dem.where(hasdata)

The cleanest way however, is by using one of the bitwise operators, inversion:

.. code-block:: python

   below_sea_level = dem < 0.0
   not_below_sea_level = ~below_sea_level
   corrected_dem = dem.where(not_below_sea_level, other=0.0)

Step by step:

* First we compare the ``dem`` to 0.0. This means nodata values end up
  ``False`` in ``below_sea_level``.
* Next we invert below_sea_level (flip around ``True`` and ``False``. This
  means values larger than 0.0, and the nodata values are marked by ``True``.
* This is the appropriate selection condition by which the other values will
  be replaced correctly by the threshold value.

The default value of ``other`` is ``np.nan``, which also acts as the default
nodata value in xarray. This plays well with xarray's reduction functions which
(unlike numpy) ignore ``np.nan`` values (thereby matching the behaviour of the
``numpy.nan``- functions such as ``np.nansum``.

.. code-block:: python

   concentration = xr.DataArray(np.random.rand(100, 100), coords, dims)
   above_threshold = (concentration > threshold).sum()

Note that ``other`` can also be another xarray.DataArray.

Combining
---------

Another powerful feature is combining two arrays via ``combine_first``.
``combine_first`` will automatically align two different arrays, and replace
``nan`` values in the first array by the values of the second.

Be warned however that if you're working with arrays that do not overlap
spatially, ``combine_first`` will simply concatenate dimensions:

.. code-block:: python

   da_1 = xr.DataArray(np.random.rand(10), {"x": np.arange(0.0, 10.0)}, ["x"])
   da_2 = xr.DataArray(np.random.rand(10), {"x": np.arange(20.0, 30.0)}, ["x"])
   da_combined = da_1.combine_first(da_2)

``combine_first`` combines well with ``.where``. We can easily select the relevant part
using ``.where``, and then fill the 

.. code-block:: python

   dem = dem.where(dem < 0.0, other=alternative)
   dem = dem.where(dem < 0.0)
   dem = dem.combine_first(alternative)

Broadcasting
------------

Numpy provides a feature called "broadcasting" to ease the use of many
functions.  Consider the following example, where we have a three dimensional
array which describes the thickness of a number of geological layers. We wish
to know the relative contribution of every layer to the total. In numpy, we
would do so as follows:

.. code-block:: python

   thickness = np.random.rand(4, 100, 100)
   total_thickness = np.sum(thickness, axis=0)
   relative_thickness = thickness / total_thickness

This works well. However, it only works because the shape of the numpy arrays 
matches, starting from the trailing dimensions and working forward.
* The dimensions of ``thickness`` are (4 layer, 100 rows, 100 columns)
* The dimensions of ``total_thickness`` are (100 rows, 100 columns)

This means that if we want to multiply the thickness of each layer by a certain
factor, we have to tranpose the array, multiply it, and transpose it back:

.. code-block:: python

   factor = np.array([5.0, 10.0, 5.0, 10.0])
   multiplied_thickness = (thickness.tranpose() * factor).transpose()

This is rather bothersome. Fortunately, xarray provides labelled dimension, and
can align automatically based on dimension name, rather than axis order. In the
examples below, this requires defining coordinates and dimensions first, but
these generally already exist in real-life applications.

Note that the xarray.DataArrays have been prefixed with ``da_`` for clarity.

.. code-block:: python

   coords = {"layer": [1, 2, 3, 4], "x": np.arange(100.0), "y": np.arange(100.0)}
   dims = ("layer", "y", "x")
   da_thickness = xr.DataArray(thickness, coords, dims)
   da_factor = xr.DataArray(factor, {"layer": [1, 2, 3, 4]}, ["layer"])
   da_mult_thickness = da_thickness * da_factor

Note that it's not just the infix (arithmetic) operators which support
broadcasting, ``.where`` automatically broadcasts as well!

.. code-block:: python

   z = np.array([-5.0, -15.0, -25.0, -35.0])
   da_z = xr.DataArray(z, {"layer": [1, 2, 3, 4]}, ["layer"])
   2dcoords = {"y": coords["y"], "x": coords["x"]}
   layer_bottom = xr.DataArray(np.random.rand(100, 100), 2dcoords, ("y", "x"))
   da_layer_conc = da_conc.where(da_z > layer_bottom)

Step by step:

* We create a DataArray describing the "z" dimension of the concentration
  array.
* We create the boundary of a first layer.
* We take values from ``da_conc`` (with dimension ``layer, y, x``) using da_z
  (with dimension ``layer``), and layer_bottom (with dimensions ``y, x``).
* This creates a three dimension boolean array by which we can select the
  appropriate values from ``da_conc``.

Morphology operations
---------------------

``scipy.ndimage.morphology`` provides a number of powerful morphological
operations, which can be easily combined with xarray to provide additional
versatility in modifying multi-dimensional arrays.
``scikit-image.morphology`` re-uses and expands on
``scipy.ndimage.morphology``'s functionality, and provides very rich
documentation.

A typical operation is ``binary_dilation`` or ``binary_erosion``. Consider a
(binary) mask, such as the location of land and sea, derived from a dem, as
done above. After determing the land surface, we might want to add a buffer of
one cell thick to describe the transition zone from land to sea, which we can
then proceed to use to assign a boundary condition to the model.

To identify these cells, we might use:

.. code-block:: python

   buffered = xr.full_like(is_land, False)
   buffered.values = skimage.morphology.binary_dilation(is_land.values, iterations=1)
   coastal_zone = buffered - is_land

Or using the inverse:

.. code-block:: python

   eroded = xr.full_like(is_land, False)
   eroded.values = skimage.morphology.binary_erosion(is_sea.values, iterations=1)
   coastal_zone = is_sea - eroded
   coastal_zone = coastal_zone.where(is_sea)

These operations also work in three dimensions.

``skimage.morphology`` has a few other functions to be aware of:

* ``skimage.morphology.label`` to find connected zones (also in 3D)
* ``skimage.morphology.watershed`` to find watershed basins
* ``skimage.morphology.skeletonize`` to reduce connected zones to a pixel wide line
* ``skimage.morphology.fill`` to perform flood filling on an image


Aggregating with groupby
------------------------

A common operation in geospatial analyses is aggregating one grid based on
another classification grid, for example land use.

We can do this via ``.where``:

.. code-block:: python

   landuse_classes = np.unique(landuse_classes)
   landuse_classes = landuse_classes[~np.isnan(landuse_classes)]
   
   aggregated = {} 
   for class in landuse_classes:
      aggregated[class] = grid.where(landuse_classes == class).sum(["y", "x"])

A more convenient way of doing this is by using ``.groupby``.

.. code-block:: python

   ds = xr.Dataset()
   ds["landuse"] = landuse_classes
   ds["grid"] = grid
   grouped = ds.groupby("landuse")
   aggregated = grouped.sum("stacked_yx")

Note that the reduction on the groupby object is rather non-obvious:
``stacked_yx``. To find out how coordinates have been combined, the following
provides a peek at the first group:

.. code-block:: python

   list(grouped)[0][0]


A note on performance
---------------------

These implementations are technically not the most efficient: the aggregation
takes multiple passes over the data, while in principle a single pass is
sufficient:

.. code-block:: python

   aggregated = {}
   nrow, ncol = grid.shape
   for i in range(nrow):
      for j in range(ncol):
         landuse = landuse_classes[i, j]
         aggregated[landuse] += grid[i, j]

While this code takes only a single pass over the data, it will generally be
much slower than the two alternatives above. The reason is that while they take
multiple passes, the passes are executed within compiled and fast numpy code,
rather than slow dynamic Python loops. ``imod`` also provides a fast
aggregation function, which does aggregate in a single pass.

This is true in general. Getting and setting values via ``.sel``, ``.isel``,
and ``.where`` will be generally much faster than writing loops to collect or
change elements. Where these methods are insufficiently fast, we might have
defined a specific fast implementation in ``imod``.

A short demonstration
---------------------

.. code-block:: python

   head = imod.idf.open("head*_l1.idf")
   is_max_in_time = head == head.max("time")
   max_year = head["time.year"].where(is_max_in_time).min("time")

This code snippet combines several features:

* We open the data for the first layer, for all times.
* We compute the year of every timestamp.
* We look for the heads (at every ``(x, y)``) that are the highest at that
  location.
* We select the year of the timestamp at the moment; the other values end up as
  nan.
* Finally, we reduce this DataArray over time, finding the first moment where
  the head reaches its highest values.
  
Theoretically, a highest value might be reached at more that one point in time.
This is easy to check:

.. code-block:: python

   occurrence = head["time.year"].where(head == head.max("time")).count("time")
   occurrence.plot()

