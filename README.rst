.. image:: https://dpcbuild.deltares.nl/guestAuth/app/rest/builds/buildType:(id:iMOD6_IMODPython_Windows_Tests)/statusIcon.png
   :target: https://github.com/Deltares/imod-python/commits/master/
.. image:: https://img.shields.io/pypi/l/imod
   :target: https://choosealicense.com/licenses/mit/
.. image:: https://img.shields.io/conda/vn/conda-forge/imod.svg
   :target: https://github.com/conda-forge/imod-feedstock
.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
   :target: https://pixi.sh
.. image:: https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038
   :target: https://scientific-python.org/specs/spec-0000/

The ``imod`` Python package is an open source project to make working with
MODFLOW groundwater models in Python easier. It builds on top of popular
packages such as `xarray`_, `pandas`_, `geopandas`_, `dask`_,  and `rasterio`_
to provide a versatile toolset for working with large groundwater modeling
datasets. Some of its core functionalities are:

* Preparing and modifying data from a variety of GIS, scientific, and MODFLOW
  file formats;
* Regridding, clipping, masking, and splitting MODFLOW 6 models;
* Fast writing of data to MODFLOW-based models;
* Selecting and evaluating, e.g. for time series comparison or water budgets;
* Visualizing cross sections, time series, or 3D animations.

We currently support the following MODFLOW-based kernels:

* `USGS MODFLOW 6`_, structured (DIS) and discretization by vertices (DISV)
  grids only. Not all advanced stress packages are supported (only LAK and UZF)
* `iMOD-WQ`_, which integrates SEAWAT (density-dependent
  groundwater flow) and MT3DMS (multi-species reactive transport calculations)

Development currently focuses on supporting more MODFLOW 6 functionalities.
iMOD-WQ has been sunset and will no longer be developed.

Why ``imod``?
=============

1\. Easily create grid-based model packages
-------------------------------------------

Seamlessly integrate your GIS rasters or meshes with MODFLOW 6, by using `xarray`_
and `xugrid`_ arrays, for structured and unstructured grids, respectively, to
create grid-based model packages. 

.. code-block:: python

  import imod

  # Open Geotiff with elevation data as xarray DataArray
  elevation = imod.rasterio.open("elevation.tif")

  # Create idomain grid
  layer_template = xr.DataArray([1, 1, 1], dims=('layer',), coords={'layer': [1, 2, 3]})
  idomain = layer_template * xr.ones_like(elevation).astype(int)

  # Compute bottom elevations of model layers
  layer_thickness = xr.DataArray([10.0, 20.0, 10.0], dims=('layer',), coords={'layer': [1, 2, 3]})
  bottom = elevation - layer_thickness.cumsum(dim='layer')

  # Create MODFLOW 6 DIS package
  dis_pkg = imod.mf6.StructuredDiscretization(
      idomain=idomain, top=elevation, bottom=bottom.transpose("layer", "y", "x")
  )


2\. Assign wells based on data at hand, instead of the model grid
-----------------------------------------------------------------

Assign wells based on x, y coordinates and filter screen depths, instead of
layer, row and column:

.. code-block:: python

  # Specify well locations
  x = [150_200.0, 160_800.0]
  y = [450_300.0, 460_200.0]

  # Specify well screen depths
  screen_top = [0.0, 0.0]
  screen_bottom = [-4.0, -10.0]

  # Specify flow rate, which changes over time.
  weltimes = pd.date_range("2000-01-01", "2000-01-03", freq="2D")
  well_rates_period1 = [0.5, 1.0]
  well_rates_period2 =  [2.5, 3.0]
  rate = xr.DataArray([well_rates_period1, well_rates_period2], coords={"time": weltimes}, dims=("time","index"))

  # Now construct the Well package
  wel_pkg = imod.mf6.Well(x=x, y=y, rate=rate, screen_top=screen_top, screen_bottom=screen_bottom)

iMOD Python will take care of the rest and assign the wells to the correct model
layers upon writing the model. It will furthermore distribute well rates based
on transmissivities. To verify how wells will be assigned to MODFLOW 6 cells before
writing the entire simulation, you can use the following command:

.. code-block:: python

  # Wells have been distributed across two model layers based on screen depths.
  wel_mf6_pkg = wel_pkg.to_mf6(idomain, top, bottom, k=1.0)
  print(wel_mf6_pkg["cellid"])

  # Well rates have been distributed based on screen overlap
  print(wel_mf6_pkg["rate"])


3\. Utilities to assign 2D river grids to 3D model layers
---------------------------------------------------------

A common problem in groundwater modeling is to assign 2D river or drain grids to
3D model layers. iMOD Python has utilities to do this, supporting all kinds of
different methods. Furthermore, it can help you distribute the conductance
across layers.

`See examples here <https://deltares.github.io/imod-python/user-guide/09-topsystem.html>`_

4\. Create stress periods based on times assigned to boundary conditions
--------------------------------------------------------------------------

MODFLOW 6 requires that all stress periods are defined in the time discretization
package. However, usually boundary conditions are defined at inconsistent
times. iMOD Python can help you to create a time discretization package that is
consistent, based on all the unique times assigned to the boundary conditions.

`See futher explanation here <https://deltares.github.io/imod-python/user-guide/07-time-discretization.html>`_

.. code-block:: python

  # First add the packages to the simulation. NOTE: To get a functional model,
  # more packages are needed than these two.
  simulation = imod.mf6.Modflow6Simulation("example")
  simulation["gwf"] = imod.mf6.GroundwaterFlowModel()
  simulation["gwf"]["dis"] = dis_pkg
  simulation["gwf"]["wel"] = wel_pkg

  # Create a time discretization based on the times assigned to the packages.
  # Specify the end time of the simulation as one of the additional_times
  simulation.create_time_discretization(additional_times=["2000-01-07"])

  # Note that timesteps in well package are also inserted in the time
  # discretization
  print(simulation["time_discretization"].dataset)


5\. Regridding MODFLOW 6 models to different grids
--------------------------------------------------

Regrid MODFLOW 6 models to different grids, even from structured to unstructured
grids. iMOD Python takes care of properly scaling the input parameters. You can
also configure scaling methods yourself for each input parameter, for example
when you want to upscale drainage elevations with the minimum instead of the
average.

.. code-block:: python

  sim_regridded = simulation.regrid_like(new_unstructured_grid)
  # Notice that discretization has converted to VerticesDiscretization (DISV)
  print(sim_regridded["gwf"]["dis"])


`See further explanation here <https://deltares.github.io/imod-python/user-guide/08-regridding.html>`_

6\. Clip MODFLOW 6 models to a bounding box
-------------------------------------------

To reduce the size of your model, you can clip it to a bounding box. This is
useful for example when you want to create a smaller model for testing purposes.

.. code-block:: python

  sim_clipped = simulation.clip_box(x_min=125_000, x_max=175_000, y_min=425_000, y_max=475_000)

You can even provide states for the model, which will be set on the model boundaries of the clipped model.

.. code-block:: python

  # Create a grid of zeros, which will be used to
  # set as heads at the boundaries of clipped parts.
  head_for_boundary = xr.zeros_like(idomain, dtype=float)
  states_for_boundary = {"gwf": head_for_boundary}

  sim_clipped = simulation.clip_box(
      x_min=125_000, x_max=175_000, y_min=425_000, y_max=475_000, states_for_boundary=states_for_boundary
  )

  # Notice that a Constant Head (CHD) package has been created for the clipped
  # model.
  print(sim_clipped["gwf"])

7\. Performant writing of MODFLOW 6 models
------------------------------------------

iMOD Python efficiently writes MODFLOW 6 models to disk, especially large models.
Tests we have conducted for the Dutch National Groundwater Model (LHM) show that
iMOD Python can write a model with 21.84 million cells 5 to 60 times faster (for
respectively 1 and 365 stress periods) than the alternative `Flopy`_ package. 
Furthermore ``imod`` can even write models that are larger than the available
memory, using `dask`_ arrays.

*NOTE:* iMOD developers work alongside the Flopy development team and contribute
to both projects.

8\. Import your iMOD5 models
----------------------------

Models made with `iMOD5`_ can be imported into iMOD Python, provided that they are
defined in a projectfile.

.. code-block:: python

  # Open projectfile data
  imod5_data, period_data = imod.formats.prj.open_projectfile_data("path/to/projectfile.prj")

  # Specify times for the simulation, this will be used to resample iMOD5 wells
  # to and to set the time discretization
  times = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02"), np.datetime64("2000-01-03")]
  
  # Create a simulation object
  simulation = imod.mf6.Modflow6Simulation.from_imod5_data(imod5_data, period_data, times)

`See this page for a full list of supported iMO5 functionalities. <https://deltares.github.io/imod-python/faq/imod5_backwards_compatibility.html>`_

Why not ``imod``?
=================

1\. You want to make a small, synthetic model
---------------------------------------------

If you are not interested in deriving models from spatial data, but just want to
allocate boundary conditions based on layer, row, column numbers, or create a
model of a 2D cross-section: You are better off using `Flopy`_. If you want to
complexify this model with a lot of stress periods and run into slow writing
speeds, consider using ``imod`` for performance.

2\. Not all MODFLOW 6 features are supported
--------------------------------------------

Currently, we don't support the following MODFLOW 6 features:

- timeseries files
- DISU package
- Groundwater Energy Model (GWE)
- Streamflow routing (SFR) package (`in development <https://github.com/Deltares/imod-python/pull/1497>`_)
- Ghost Node Correction (GNC) package
- Multi-aquifer well (MAW) package
- Water mover (MVR) package
- Particle tracking (PRT)

Most of these features can be implemented with some effort, but we have not
prioritized them yet. The exceptions are the DISU package and the timeseries
files, which would require significant work to our backend. As a result, we will
likely not support these two features in the foreseeable future. If you need any of the
other features, feel free to open an issue on our GitHub page.

Additional links
================

Documentation: https://deltares.github.io/imod-python

Source code: https://github.com/Deltares/imod-python

Issues: https://github.com/Deltares/imod-python/issues

.. _Deltares: https://www.deltares.nl
.. _dask: https://dask.org/
.. _xarray: http://xarray.pydata.org/
.. _xugrid: https://deltares.github.io/xugrid/
.. _pandas: http://pandas.pydata.org/
.. _rasterio: https://rasterio.readthedocs.io/en/latest/
.. _geopandas: http://geopandas.org/
.. _netCDF: https://www.unidata.ucar.edu/software/netcdf/
.. _USGS MODFLOW 6: https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model
.. _iMOD-WQ: https://oss.deltares.nl/web/imod
.. _iMOD5: https://oss.deltares.nl/web/imod
.. _Flopy: https://flopy.readthedocs.io/en/latest/
