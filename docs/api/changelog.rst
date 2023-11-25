Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.


[Unreleased]
------------

Fixed
~~~~~
- The Newton option for a :class:`imod.mf6.GroundwaterFlowModel` was being ignored. This has been
  corrected.
- The Contextily packages started throwing errors. This was caused because the
  default tile provider being used was Stamen. However Stamen is no longer free
  which caused Contextily to fail. The default tile provider has been changed to
  OpenStreetMap to resolve this issue.
- :function:`imod.mf6.open_cbc` now reads saved cell saturations and specific discharges.
- :function:`imod.mf6.open_cbc` failed to read unstructured budgets stored
  following IMETH1, most importantly the storage fluxes.
- Fixed support of Python 3.11 by dropping the obsolete ``qgs`` module.
- Bug in :class:`imod.mf6.SourceSinkMixing` where, in case of multiple active
  boundary conditions with assigned concentrations, it would write a ``.ssm``
  file with all sources/sinks on one single row.

Changed
~~~~~~~
- The imod-environment.yml file has been split in an imod-environment.yml
  (containing all packages required to run imod-python) and a
  imod-environment-dev.yml file (containing additional packages for developers).
- Changed the way :class:`imod.mf6.Modflow6Simulation`,
  :class:`imod.mf6.GroundwaterFlowModel`,
  :class:`imod.mf6.GroundwaterTransportModel`, and Modflow 6 packages are
  represented while printing.

Added
~~~~~
- The unit tests results are now published on GitLab
- A ``save_saturation`` option to :class:`imod.mf6.NodePropertyFlow` which saves
  cell saturations for unconfined flow.
- Functions :function:`imod.prepare.layer.get_upper_active_layer_number` and
  :function:`imod.prepare.layer.get_lower_active_layer_number` to return planar
  grids with numbers of the highest and lowest active cells respectively.
- Functions :function:`imod.prepare.layer.get_upper_active_grid_cells` and
  :function:`imod.prepare.layer.get_lower_active_grid_cells` to return boolean
  grids designating respectively the highest and lowest active cells in a grid.
- validation of ``transient`` argument in :class:`imod.mf6.StorageCoefficient`
  and :class:`imod.mf6.SpecificStorage`.
- :meth:`imod.mf6.Simulation.open_concentration`,
  :meth:`imod.mf6.Simulation.open_head`,
  :meth:`imod.mf6.Simulation.open_transport_budget`, and
  :meth:`imod.mf6.Simulation.open_flow_budget`, were added as convenience
  methods to open simulation output easier (without having to specify paths).
- The :meth:`imod.mf6.Simulation.split` method has been added. This method makes
  it possible for a user to create a Multi-Model simulation. A user needs to
  provide a submodel label array in which they specify to which submodel a cell
  belongs. The method will then create the submodels and split the nested
  packages. The split method will create the gwfgwf exchanges required to
  connect the submodels. At the moment auxiliary variables ``cdist`` and
  ``angldegx`` are only computed for structured grids. 
- The label array can be generated through a convenience function
  :function:`imod.mf6.partition_generator.get_label_array`
- Once a split simulation has been executed by MF6, we find head and balance
  results in each of the partition models. These can now be merged into head and
  balance datasets for the original domain using
  :meth:`imod.mf6.Simulation.open_concentration`,
  :meth:`imod.mf6.Simulation.open_head`,
  :meth:`imod.mf6.Simulation.open_transport_budget`,
  :meth:`imod.mf6.Simulation.open_flow_budget`.
  In the case of balances, the exchanges through the partition boundary are not
  yet added to this merged balance. 
- Settings such as ``save_flows`` can be passed through
  :meth:`imod.mf6.SourceSinkMixing.from_flow_model`

Removed
~~~~~~~
- Tox has been removed from the project.
- Dropped support for writing .qgs files directly for QGIS, as this was hard to
  maintain and rarely used. To export your model to QGIS readable files, call
  the ``dump`` method :class:`imod.mf6.Simulation` with ``mdal_compliant=True``.
  This writes UGRID NetCDFs which can read as meshes in QGIS.
- Removed ``declxml`` from repository.

[0.14.1] - 2023-09-07
---------------------

Changed
~~~~~~~

- TWRI Modflow 6 example uses the grid-agnostic :class:`imod.mf6.Well`
  package instead of the :class:`imod.mf6.WellDisStructured` package.

Fixed
~~~~~

- :class:`imod.mf6.HorizontalFlowBarrier` would write to a binary file by
  default. However, the current version of Modflow 6 does not support this.
  Therefore, this class now always writes to text file.


[0.14.0] - 2023-09-06
---------------------

Changed
~~~~~~~

- :class:`imod.mf6.HorizontalFlowBarrier` is specified by providing a geopandas
  `GeoDataFrame
  <https://geopandas.org/en/stable/docs/reference/geodataframe.html>`_


Added
~~~~~

- :meth:`imod.mf6.Simulation.regrid_like` to regrid a Modflow6 simulation to a
  new grid (structured or unstructured), using `xugrid's regridding
  functionality.
  <https://deltares.github.io/xugrid/examples/regridder_overview.html>`_
  Variables are regridded with pre-selected methods. The regridding
  functionality is useful for a variety of applications, for example to test the
  effect of different grid sizes, to add detail to a simulation (by refining the
  grid) or to speed up a simulation (by coarsening the grid) to name a few
- :meth:`imod.mf6.Package.regrid_like` to regrid packages. The user can
  specify their own custom regridder types and methods for variables.
- :meth:`imod.mf6.Simulation.clip_box` got an extra argument
  ``states_for_boundary``, which takes a dictionary with modelname as key and
  griddata as value. This data is specified as fixed state on the model
  boundary. At present only `imod.mf6.GroundwaterFlowModel` is supported, grid
  data is specified as a :class:`imod.mf6.ConstantHead` at the model boundary.
- :class:`imod.mf6.Well`, a grid-agnostic well package, where wells can be
  specified based on their x,y coordinates and filter top and bottom.


[0.13.2] - 2023-07-26
---------------------

Changed
~~~~~~~

- :func:`imod.rasterio.save` will now write ESRII ASCII rasters, even if
  rasterio is not installed. A fallback function has been added specifically
  for ASCII rasters.

Fixed
~~~~~

- Geopandas and rasterio were imported at the top of a module in some places.
  This has been fixed so that both are not optional dependencies when
  installing via pip (installing via conda or mamba will always pull all
  dependencies and supports full functionality).
- :meth:`imod.mf6.Simulation._validate` now print all validation errors for all
  models and packages in one message.
- The gen file reader can now handle feature id's that contain commas and spaces
- :class:`imod.mf6.EvapoTranspiration` now supports segments, by adding a
  ``segment`` dimension to the ``proportion_depth`` and ``proportion_rate``
  variables.
- :class:`imod.mf6.EvapoTranspiration` template for ``.evt`` file now properly 
  formats ``nseg`` option.
- Fixed bug in :class:`imod.wq.Well` preventing saving wells without a time
  dimension, but with a layer dimension.
- :class:`imod.mf6.DiscretizationVertices._validate` threw ``KeyError`` for
  ``"bottom"`` when validating the package separately.

Added
~~~~~

- :func:`imod.select.grid.active_grid_boundary_xy` & 
  :func:`imod.select.grid.grid_boundary_xy` are added to find grid boundaries.

[0.13.1] - 2023-05-05
---------------------

Added
~~~~~

- :class:`imod.mf6.SpecificStorage` and :class:`imod.mf6.StorageCoefficient` 
  now have a ``save_flow`` argument.

Fixed
~~~~~

- :func:`imod.mf6.open_cbc` can now read storage fluxes without error.


[0.13.0] - 2023-05-02
---------------------

Added
~~~~~

- :class:`imod.mf6.OutputControl` now takes parameters ``head_file``,
  ``concentration_file``, and ``budget_file`` to specify where to store
  MODFLOW6 output files.
- :func:`imod.util.from_mdal_compliant_ugrid2d` to "restack" the variables that
  have have been "unstacked" in :func:`imod.util.mdal_compliant_ugrid2d`.
- Added support for the Modflow6 Lake package
- :func:`imod.select.points_in_bounds`, :func:`imod.select.points_indices`,
  :func:`imod.select.points_values` now support unstructured grids.
- Added support for the Modflow 6 Lake package: :class:`imod.mf6.Lake`,
  :class:`imod.mf6.LakeData`, :class:`imod.mf6.OutletManning`, :class:`OutletSpecified`,
  :class:`OutletWeir`. See the examples for an application of the Lake package.
- :meth:`imod.mf6.simulation.Modflow6Simulation.dump` now supports dumping to MDAL compliant
  ugrids. These can be used to view and explore Modlfow 6 simulations in QGIS.

Fixed
~~~~~

- :meth:`imod.wq.bas.BasicFlow.thickness` returns a DataArray with the correct
  dimension order again. This confusingly resulted in an error when writing the
  :class:`imod.wq.btn.BasicTransport` package.
- Fixed bug in :class:`imod.mf6.dis.StructuredDiscretization` and
  :class:`imod.mf6.dis.VerticesDiscretization` where 
  ``inactive bottom above active cell`` was incorrectly raised.

[0.12.0] - 2023-03-17
---------------------

Added
~~~~~

- :func:`imod.prj.read_projectfile` to read the contents of a project file into
  a Python dictionary.
- :func:`imod.prj.open_projectfile_data` to read/open the data that is pointed
  to in a project file.
- :func:`imod.gen.read_ascii` to read the geometry stored in ASCII text .gen files.
- :class:`imod.mf6.hfb.HorizontalFlowBarrier` to support Modflow6's HFB
  package, works well with `xugrid.snap_to_grid` function.
- :meth:`imod.mf6.simulation.Modflow6Simulation.dump` to dump a simulation to a toml file
  which acts as a definition file, pointing to packages written as netcdf files. This
  can be used to intermediately store Modflow6 simulations.

Fixed
~~~~~

- :func:`imod.evaluate.budget.flow_velocity` now properly computes velocity by
  dividing by the porosity. Before, this function computed the Darcian velocity.

Changed
~~~~~~~

- :func:`imod.ipf.save` will error on duplicate IDs for associated files if a
  ``"layer"`` column is present. As a dataframe is automatically broken down
  into a single IPF per layer, associated files for the first layer would be
  overwritten by the second, and so forth.
- :meth:`imod.wq.Well.save` will now write time varying data to associated
  files for extration rate and concentration.
- Choosing ``method="geometric_mean"`` in the Regridder will now result in NaN
  values in the regridded result if a geometric mean is computed over negative
  values; in general, a geometric mean should only be computed over physical
  quantities with a "true zero" (e.g. conductivity, but not elevation).

[0.11.6] - 2023-02-01
---------------------

Added
~~~~~

- Added an extra optional argument in
  :meth:`imod.couplers.metamod.MetaMod.write` named ``modflow6_write_kwargs``,
  which can be used to provide keyword arguments to the writing of the Modflow 6
  Simulation.

Fixed
~~~~~

- :func:`imod.mf6.out.disv.read_grb` Remove repeated construction of
  ``UgridDataArray`` for ``top``

[0.11.5] - 2022-12-15
---------------------

Fixed
~~~~~

- :meth:`imod.mf6.Modflow6Simulation.write` with ``binary=False`` no longer
  results in invalid MODFLOW6 input for 2D grid data, such as DIS top.
- :meth:`imod.flow.ImodflowModel.write` no longer writes incorrect project
  files for non-grid values with a time and layer dimension.
- :func:`imod.evaluate.interpolate_value_boundaries`: Fix edge case when
  successive values in z direction are exactly equal to the boundary value.

Changed
~~~~~~~

- Removed ``meshzoo`` dependency.
- Minor changes to :mod:`imod.gen.gen` backend, to support `Shapely 2.0
  <https://shapely.readthedocs.io/en/latest/release/2.x.html>`_ , Shapely
  version above equal v1.8 is now required.

Added
~~~~~

- :meth:`imod.flow.ImodflowModel.write` now supports writing a
  ``config_run.ini`` to convert the projectfile to a runfile or modflow 6
  namfile with iMOD5.
- Added validation of Modflow6 Flow and Transport models. Incorrect model input
  will now throw a ``ValidationError``. To turn off the validation, set
  ``validate=False`` upon package initialization and/or when calling
  :meth:`imod.mf6.Modflow6Simulation.write`.

[0.11.4] - 2022-09-05
---------------------

Fixed
~~~~~

- :meth:`imod.mf6.GroundwaterFlowModel.write` will no longer error when a 3D
  DataArray with a single layer is written. It will now accept both 2D and 3D
  arrays with a single layer coordinate.
- Hotfixes for :meth:`imod.wq.model.SeawatModel.clip`, until `this merge request
  <https://gitlab.com/deltares/imod/imod-python/-/merge_requests/111>`_ is
  fulfilled. 
- :meth:`imod.flow.ImodflowModel.write` will set the timestring in the
  projectfile to ``steady-state`` for ``BoundaryConditions`` without a time
  dimension.
- Added :class:`imod.flow.OutputControl` as this was still missing.
- :func:`imod.ipf.read` will no longer error when an associated files with 0
  rows is read.
- :func:`imod.evaluate.calculate_gxg` now correctly uses (March 14, March
  28, April 14) to calculate GVG rather than (March 28, April 14, April 28).
- :func:`imod.mf6.out.open_cbc` now correctly loads boundary fluxes.
- :meth:`imod.prepare.LayerRegridder.regrid` will now correctly skip values
  if ``top_source`` or ``bottom_source`` are NaN.
- :func:`imod.gen.write` no longer errors on dataframes with empty columns.

Changed
~~~~~~~

- Deprecate the current documentation URL: https://imod.xyz. For the coming
  months, redirection is automatic to:
  https://deltares.gitlab.io/imod/imod-python/.
- :func:`imod.ipf.save` will now store associated files in separate directories
  named ``layer1``, ``layer2``, etc. The ID in the main IPF file is updated
  accordingly. Previously, if IDs were shared between different layers, the
  associated files would be overwritten as the IDs would result in the same
  file name being used over and over.
- :meth:`imod.flow.ImodflowModel.time_discretization`,
  :meth:`imod.wq.SeawatModel.time_discretization`,
  :meth:`imod.mf6.Simulation.time_discretization`,
  are renamed to:
  :meth:`imod.flow.ImodflowModel.create_time_discretization`,
  :meth:`imod.wq.SeawatModel.create_time_discretization`,
  :meth:`imod.mf6.Simulation.create_time_discretization`,
- Moved tests inside `imod` directory, added an entry point for pytest fixtures.
  Running the tests now requires an editable install, and also existing
  installations have to be reinstalled to run the tests.
- The ``imod.mf6`` model packages now all run type checks on input. This is a
  breaking change for scripts which provide input with an incorrect dtype.
- :class:`imod.mf6.Solution` now requires a `model_names` argument to specify
  which models should be solved in a single numerical solution. This is
  required to simulate groundwater flow and transport as they should be
  in separate solutions.
- When writing MODFLOW6 input option blocks, a NaN value is now recognized as
  an alternative to None (and the entry will not be included in the options
  block).

Added
~~~~~

- Added support to write MetaSWAP models, :class:`imod.msw.MetaSwapModel`.
- Addes support to write coupled MetaSWAP and Modflow6 simulations,
  :class:`imod.couplers.MetaMod`
- :func:`imod.util.replace` has been added to find and replace different values
  in a DataArray.
- :func:`imod.evaluate.calculate_gxg_points` has been added to compute GXG
  values for time varying point data (i.e. loaded from IPF and presented as a
  Pandas dataframe).
- :func:`imod.evaluate.calculate_gxg` will return the number of years used
  in the GxG calculation as separate variables in the output dataset.
- :func:`imod.visualize.spatial.plot_map` now accepts a `fix` and `ax` argument,
  to enable adding maps to existing axes.
- :meth:`imod.flow.ImodflowModel.create_time_discretization`,
  :meth:`imod.wq.SeawatModel.create_time_discretization`,
  :meth:`imod.mf6.Simulation.create_time_discretization`, now have a
  documentation section.
- :class:`imod.mf6.GroundwaterTransportModel` has been added with associated
  simple classes to allow creation of solute transport models. Advanced
  boundary conditions such as LAK or UZF are not yet supported.
- :class:`imod.mf6.Buoyancy` has been added to simulate density dependent
  groundwater flow.

[0.11.1] - 2021-12-23
---------------------

Fixed
~~~~~

-  ``contextily``, ``geopandas``, ``pyvista``, ``rasterio``, and ``shapely``
   are now fully optional dependencies. Import errors are only raised when
   accessing functionality that requires their use.
-  Include declxml as ``imod.declxml`` (should be internal use only!): declxml
   is no longer maintained on the official repository:
   https://github.com/gatkin/declxml. Furthermore, it has no conda feedstock,
   which makes distribution via conda difficult.

[0.11.0] - 2021-12-21
---------------------

Fixed
~~~~~

-  :func:`imod.ipf.read` accepts list of file names.
-  :func:`imod.mf6.open_hds` did not read the appropriate bytes from the
   heads file, apart for the first timestep. It will now read the right records.
-  Use the appropriate array for modflow6 timestep duration: the
   :meth:`imod.mf6.GroundwaterFlowModel.write` would write the timesteps
   multiplier in place of the duration array.
-  :meth:`imod.mf6.GroundwaterFlowModel.write` will now respect the layer
   coordinate of DataArrays that had multiple coordinates, but were
   discontinuous from 1; e.g. layers [1, 3, 5] would've been transformed to [1,
   2, 3] incorrectly.
-  :meth:`imod.mf6.Modflow6Simulation.write` will no longer change working directory
   while writing model input -- this could lead to errors when multiple
   processes are writing models in parallel.
-  :func:`imod.prepare.laplace_interpolate` will no longer ZeroDivisionError
   when given a value for ``ibound``.

Added
~~~~~

-  :func:`imod.idf.open_subdomains` will now also accept iMOD-WQ output of
   multiple species runs.
-  :meth:`imod.wq.SeawatModel.to_netcdf` has been added to write all model
   packages to netCDF files.
-  :func:`imod.mf6.open_cbc` has been added to read the budget data of
   structured (DIS) MODFLOW6 models. The data is read lazily into xarray
   DataArrays per timestep.
-  :func:`imod.visualize.streamfunction` and :func:`imod.visualize.quiver`
   were added to plot a 2D representation of the groundwater flow field using
   either streamlines or quivers over a cross section plot
   (:func:`imod.visualize.cross_section`).
-  :func:`imod.evaluate.streamfunction_line` and
   :func:`imod.evaluate.streamfunction_linestring` were added to extract the
   2D projected streamfunction of the 3D flow field for a given cross section.
-  :func:`imod.evaluate.quiver_line` and :func:`imod.evaluate.quiver_linestring`
   were added to extract the u and v components of the 3D flow field for a given
   cross section.
-  Added :meth:`imod.mf6.GroundwaterFlowModel.write_qgis_project` to write a
   QGIS project for easier inspection of model input in QGIS.
-  Added :meth:`imod.wq.SeawatModel.clip` to clip a model to a provided extent.
   Boundary conditions of clipped model can be automatically derived from parent
   model calculation results and are applied along the edges of the extent.
-  Added :py:func:`imod.gen.read` and :py:func:`imod.gen.write` for reading
   and writing binary iMOD GEN files to and from geopandas GeoDataFrames.
-  Added :py:func:`imod.prepare.zonal_aggregate_raster` and
   :py:func:`imod.prepare.zonal_aggregate_polygons` to efficiently compute zonal
   aggregates for many polygons (e.g. the properties every individual ditch in
   the Netherlands).
-  Added :py:class:`imod.flow.ImodflowModel` to write to model iMODFLOW project
   file.
-  :meth:`imod.mf6.Simulation.write` now has a ``binary`` keyword. When set
   to ``False``, all MODFLOW6 input is written to text rather than binary files.
-  Added :class:`imod.mf6.DiscretizationVertices` to write MODFLOW6 DISV model
   input.
-  Packages for :class:`imod.mf6.GroundwaterFlowModel` will now accept
   :class:`xugrid.UgridDataArray` objects for (DISV) unstructured grids, next to
   :class:`xarray.DataArray` objects for structured (DIS) grids.
-  Transient wells are now supported in :class:`imod.mf6.WellDisStructured` and
   :class:`imod.mf6.WellDisVertices`.
-  :func:`imod.util.to_ugrid2d` has been added to convert a (structured) xarray
   DataArray or Dataset to a quadrilateral UGRID dataset.
-  Functions created to create empty DataArrays with greater ease:
   :func:`imod.util.empty_2d`, :func:`imod.util.empty_2d_transient`,
   :func:`imod.util.empty_3d`, and :func:`imod.util.empty_3d_transient`.
-  :func:`imod.util.where` has been added for easier if-then-else operations,
   especially for preserving NaN nodata values.
-  :meth:`imod.mf6.Simulation.run` has been added to more easily run a model,
   especially in examples and tests.
-  :func:`imod.mf6.open_cbc` and :func:`imod.mf6.open_hds` will automatically
   return a ``xugrid.UgridDataArray`` for MODFLOW6 DISV model output.

Changed
~~~~~~~

-  Documentation overhaul: different theme, add sample data for examples, add
   Frequently Asked Questions (FAQ) section, restructure API Reference. Examples
   now ru
-  Datetime columns in IPF associated files (via
   :func:`imod.ipf.write_assoc`) will not be placed within quotes, as this can
   break certain iMOD batch functions.
-  :class:`imod.mf6.Well` has been renamed into :class:`imod.mf6.WellDisStructured`.
-  :meth:`imod.mf6.GroundwaterFlowModel.write` will now write package names
   into the simulation namefile.
-  :func:`imod.mf6.open_cbc` will now return a dictionary with keys
   ``flow-front-face, flow-lower-face, flow-right-face`` for the face flows,
   rather than ``front-face-flow`` for better consistency.
-  Switched to composition from inheritance for all model packages: all model
   packages now contain an internal (xarray) Dataset, rather than inheriting
   from the xarray Dataset.
-  :class:`imod.mf6.SpecificStorage` or :class:`imod.mf6.StorageCoefficient` is
   now mandatory for every MODFLOW6 model to avoid accidental steady-state
   configuration.

Removed
~~~~~~~

-  Module ``imod.tec`` for reading Tecplot files has been removed.

[0.10.1] - 2020-10-19
---------------------

Changed
~~~~~~~

-  :meth:`imod.wq.SeawatModel.write` now generates iMOD-WQ runfiles with
   more intelligent use of the "macro tokens". ``:`` is used exclusively for
   ranges; ``$`` is used to signify all layers. (This makes runfiles shorter,
   speeding up parsing, which takes a significant amount of time in the runfile
   to namefile conversion of iMOD-WQ.)
-  Datetime formats are inferred based on length of the time string according to
   ``%Y%m%d%H%M%S``; supported lengths 4 (year only) to 14 (full format string).

Added
~~~~~

-  :class:`imod.wq.MassLoading` and
   :class:`imod.wq.TimeVaryingConstantConcentration` have been added to allow
   additional concentration boundary conditions.
-  IPF writing methods support an ``assoc_columns`` keyword to allow greater
   flexibility in including and renaming columns of the associated files.
-  Optional basemap plotting has been added to :meth:`imod.visualize.plot_map`.

Fixed
~~~~~

-  IO methods for IDF files will now correctly identify double precision IDFs.
   The correct record length identifier is 2295 rather than 2296 (2296 was a
   typo in the iMOD manual).
-  :meth:`imod.wq.SeawatModel.write` will now write the correct path for
   recharge package concentration given in IDF files. It did not prepend the
   name of the package correctly (resulting in paths like
   ``concentration_l1.idf`` instead of ``rch/concentration_l1.idf``).
-  :meth:`imod.idf.save` will simplify constant cellsize arrays to a scalar
   value -- this greatly speeds up drawing in the iMOD-GUI.

[0.10.0] - 2020-05-23
---------------------

Changed
~~~~~~~

-  :meth:`imod.wq.SeawatModel.write` no longer automatically appends the model
   name to the directory where the input is written. Instead, it simply writes
   to the directory as specified.
-  :func:`imod.select.points_set_values` returns a new DataArray rather than
   mutating the input ``da``.
-  :func:`imod.select.points_values` returns a DataArray with an index taken
   from the data of the first provided dimensions if it is a ``pandas.Series``.
-  :meth:`imod.wq.SeawatModel.write` now writes a runfile with ``start_hour``
   and ``start_minute`` (this results in output IDFs with datetime format
   ``"%Y%m%d%H%M"``).

Added
~~~~~

-  :meth:`from_file` constructors have been added to all `imod.wq.Package`.
   This allows loading directly package from a netCDF file (or any file supported by
   ``xarray.open_dataset``), or a path to a Zarr directory with suffix ".zarr" or ".zip".
-  This can be combined with the `cache` argument in :meth:`from_file` to
   enable caching of answers to avoid repeated computation during
   :meth:`imod.wq.SeawatModel.write`; it works by checking whether input and
   output files have changed.
-  The ``resultdir_is_workspace`` argument has been added to :meth:`imod.wq.SeawatModel.write`.
   iMOD-wq writes a number of files (e.g. list file) in the directory where the
   runfile is located. This results in mixing of input and output. By setting it
   ``True``, **all** model output is written in the results directory.
-  :func:`imod.visualize.imshow_topview` has been added to visualize a complete
   DataArray with atleast dimensions ``x`` and ``y``; it dumps PNGs into a
   specified directory.
-  Some support for 3D visualization has been added.
   :func:`imod.visualize.grid_3d` and :func:`imod.visualize.line_3d` have been
   added to produce ``pyvista`` meshes from ``xarray.DataArray``'s and
   ``shapely`` polygons, respectively.
   :class:`imod.visualize.GridAnimation3D` and :class:`imod.visualize.StaticGridAnimation3D`
   have been added to setup 3D animations of DataArrays with transient data.
-  Support for out of core computation by ``imod.prepare.Regridder`` if ``source``
   is chunked.
-  :func:`imod.ipf.read` now reports the problematic file if reading errors occur.
-  :func:`imod.prepare.polygonize` added to polygonize DataArrays to GeoDataFrames.
-  Added more support for multiple species imod-wq models, specifically: scalar concentration
   for boundary condition packages and well IPFs.

Fixed
~~~~~

-  :meth:`imod.prepare.Regridder` detects if the ``like`` DataArray is a subset
   along a dimension, in which case the dimension is not regridded.
-  :meth:`imod.prepare.Regridder` now slices the ``source`` array accurately
   before regridding, taking cell boundaries into account rather than only
   cell midpoints.
-  ``density`` is no longer an optional argument in :class:`imod.wq.GeneralHeadboundary` and
   :class:`imod.wq.River`. The reason is that iMOD-WQ fully removes (!) these packages if density
   is not present.
-  :func:`imod.idf.save` and :func:`imod.rasterio.save` will now also save DataArrays in
   which a coordinate other than ``x`` or ``y`` is descending.
-  :func:`imod.visualize.plot_map` enforces decreasing ``y``, which ensures maps are not plotted
   upside down.
-  :func:`imod.util.coord_reference` now returns a scalar cellsize if coordinate is equidistant.
-  :meth:`imod.prepare.Regridder.regrid` returns cellsizes as scalar when coordinates are
   equidistant.
-  Raise proper ValueError in :meth:`imod.prepare.Regridder.regrid` consistenly when the number
   of dimensions to regrid does not match the regridder dimensions.
-  When writing DataArrays that have size 1 in dimension ``x`` or ``y``: raise error if cellsize
   (``dx`` or ``dy``) is not specified; and actually use ``dy`` or ``dx`` when size is 1.

[0.9.0] - 2020-01-19
--------------------

Added
~~~~~

-  IDF files representing data of arbitrary dimensionality can be opened and
   saved. This enables reading and writing files with more dimensions than just x,
   y, layer, and time.
-  Added multi-species support for (:mod:`imod.wq`)
-  GDAL rasters representing N-dimensional data can be opened and saved similar to (:mod:`imod.idf`) in (:mod:`imod.rasterio`)
-  Writing GDAL rasters using :meth:`imod.rasterio.save` and (:meth:`imod.rasterio.write`) auto-detects GDAL driver based on file extension
-  64-bit IDF files can be opened :meth:`imod.idf.open`
-  64-bit IDF files can be written using :meth:`imod.idf.save` and (:meth:`imod.idf.write`) using keyword ``dtype=np.float64``
-  ``sel`` and ``isel`` methods to ``SeawatModel`` to support taking out a subdomain
-  Docstrings for the Modflow 6 classes in :mod:`imod.mf6`
-  :meth:`imod.select.upper_active_layer` function to get the upper active layer from ibound ``xr.DataArray``

Changed
~~~~~~~

-  :func:`imod.idf.read` is deprecated, use :mod:`imod.idf.open` instead
-  :func:`imod.rasterio.read` is deprecated, use :mod:`imod.rasterio.open` instead

Fixed
~~~~~

-  :meth:`imod.prepare.reproject` working instead of silently failing when given a ``"+init=ESPG:XXXX`` CRS string

[0.8.0] - 2019-10-14
--------------------

Added
~~~~~
-  Laplace grid interpolation :meth:`imod.prepare.laplace_interpolate`
-  Experimental Modflow 6 structured model write support :mod:`imod.mf6`
-  More supported visualizations :mod:`imod.visualize`
-  More extensive reading and writing of GDAL raster in :mod:`imod.rasterio`

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
