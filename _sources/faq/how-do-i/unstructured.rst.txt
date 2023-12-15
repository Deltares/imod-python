Unstructured
------------

imod uses `xugrid`_ to represent data on unstructured grids. Where possible,
the functions of methods of xugrid match those xarray. However, working with
the unstructured topology requires slightly different commands: any operation
dealing the unstructured (x, y) dimensions requires the ``.ugrid`` accessor.

Generate a mesh
~~~~~~~~~~~~~~~

We've developed a separate package called `pandamesh`_ to help generated (x, y)
unstructured meshes based on vector data in the form of geopandas
GeoDataFrames.

.. code-block:: python

    import geopandas as gpd
    import pandamesh as pm
    import xugrid as xu
    
    gdf = gpd.read_file("my-study-area.shp")
    gdf["cellsize"] = 100.0
    mesher = pm.TriangleMesher(gdf)
    vertices, cells = mesher.generate()
    grid = xugrid.Ugrid2d(vertices, -1, cells)

It can be installed with::

    pip install pandamesh

.. note::

    One of the dependencies of pandamesh, the Python bindings to triangle, `does
    not have the (binary) wheels for Python 3.9 and higher
    yet <https://github.com/drufat/triangle/issues/57>`_.
    
Plot a timeseries for a single cell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    transient_uda.ugrid.sel(x=x, y=y).plot()
 
Plot head of one layer at one time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    transient_uda.sel(layer=1, time="2020-01-01").ugrid.plot()

.. note::

    Since layer and time do not depend on the unstructured topology, they may
    be indexed using the standard xarray methods, without the ``.ugrid``
    accessor.

Fill/Interpolate nodata
~~~~~~~~~~~~~~~~~~~~~~~

To do Laplace interplation (using a linear equation, similar to a groundwater
model with constant conductivity):

.. code-block:: python

    uda = with_gaps.ugrid.laplace_interpolate()
    
Select points (from a vector dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    geometry = geodataframe.geometry
    x = geometry.x
    y = geometry.y
    selection = uda.ugrid.select_points(x=x, y=y)

For time series analysis, converting into a pandas DataFrame may be useful:

.. code-block:: python

    df = selection.to_dataframe()

.. _xugrid: https://deltares.github.io/xugrid/
.. _pandamesh: https://github.com/deltares/pandamesh
