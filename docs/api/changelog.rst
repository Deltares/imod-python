Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

[Unreleased]
------------

Added
~~~~~

- :meth:`imod.mf6.River.reallocate`, :meth:`imod.mf6.Drainage.reallocate`,
  :meth:`imod.mf6.GeneralHeadBoundary.reallocate`,
  :meth:`imod.mf6.Recharge.reallocate` to reallocate the package data to a new
  discretization or :class:`imod.mf6.NodePropertyFlow` package, or to use a
  different :class:`imod.prepare.ALLOCATION_OPTION` or
  :class:`imod.prepare.DISTRIBUTING_OPTION`.
- Added :meth:`imod.mf6.HorizontalFlowBarrierResistance.snap_to_grid` and
  :meth:`imod.mf6.HorizontalFlowBarrierSingleLayerResistance.snap_to_grid` to
  debug how horizontal flow barriers are snapped to a grid.

Fixed
~~~~~

- Reduce noisy warnings in models loaded with
  :meth:`imod.mf6.Modflow6Simulation.from_imod5_data` which have layers with
  cells with zero thicknesses.
- Issue where regridding lead to excessively large inactive areas.
- Issue where regridding would lead to very large negative integer values (like
  IDOMAIN) for inactive areas.
- Issue where :meth:`imod.mf6.Well.from_imod5_data` and
  :meth:`imod.mf6.LayeredWell.from_imod5_data` would throw a KeyError 0 upon
  trying to resample timeseries with a non-zero index.
- Fixed bug where :class:`imod.mf6.HorizontalFlowBarrierResistance`,
  :class:`imod.mf6.HorizontalFlowBarrierSingleLayerResistance` and other HFB
  packages would have resistances that were double the expected value with
  xugrid >= 0.14.2

Changed
~~~~~~~

- :meth:`imod.mf6.StructuredDiscretization.from_imod5_data` and
  :meth:`imod.mf6.NodePropertyFlow.from_imod5_data` now automatically load the
  dataset into memory. This improves performance when loading models with
  multiple topsystem packages.
- No upper limit anymore for ``mod_id`` in ``mod2svat.inp`` for
  :class:`imod.msw.CouplerMapping`.

Removed
~~~~~~~

- Removed ``imod.mf6.WellDisStructured`` and ``imod.mf6.WellDisVertices``. Use
  :class:`imod.mf6.Well` and :class:`imod.mf6.LayeredWell` instead. The
  :class:`imod.mf6.Well` package can be used to specify wells with filters,
  :class:`imod.mf6.LayeredWell` directly to layers.
- Removed ``imod.mf6.multimodel.partition_generator.get_label_array``, use
  :func:`imod.prepare.create_partition_labels` instead.
- Removed ``imod.idf.read`` use :func:`imod.idf.open` instead.
- Removed ``imod.rasterio.read`` use :func:`imod.rasterio.open` instead.
- Removed ``head`` argument for :class:`imod.mf6.InitialConditions`, use
  ``start`` instead.
- Removed ``cell_averaging`` argument for :class:`imod.mf6.NodePropertyFlow`,
  use ``alternative_cell_averaging`` instead.
- Removed ``set_repeat_stress`` method from boundary condition packages like
  :class:`imod.mf6.River`. Use ``repeat_stress`` argument instead.
- Removed ``time_discretization`` method from
  :class:`imod.mf6.Modflow6Simulation` and :class:`imod.wq.SeawatModel`. Use
  :meth:`imod.mf6.Modflow6Simulation.create_time_discretization` and
  :meth:`imod.wq.SeawatModel.create_time_discretization` instead.
- Removed ``imod.util.round_extent``, use :func:`imod.prepare.round_extent`
  instead.
- Removed :class:`imod.prepare.Regridder`. Use the `xugrid regridder
  <https://deltares.github.io/xugrid/examples/regridder_overview.html>`_ instead.


[1.0.0rc4] - 2025-06-20
-----------------------

Added
~~~~~

- Added ``weights`` argument to :func:`imod.prepare.create_partition_labels` to
  weigh how the simulation should be partioned. Areas with higher weights will
  result in smaller partions.
- iMOD Python version is now written in a comment line to MODFLOW6 and
  MetaSWAP's ``para_sim.inp`` files. This is useful for debugging purposes.
- Added option ``ignore_time_purge_empty`` to
  :class:`imod.mf6.Modflow6Simulation.split` to consider a package empty if its
  first times step is all nodata. This can save a lot of time splitting
  transient models.
- Add :class:`imod.mf6.ValidationSettings` to specify validation settings for
  MODFLOW 6 simulations. You can provide it to the
  :class:`imod.mf6.Modflow6Simulation` constructor.

Fixed
~~~~~

- Upon providing an unexpected coordinate in the mask or regridding grid,
  :meth:`imod.mf6.Modflow6Simulation.regrid_like` and
  :meth:`imod.mf6.Modflow6Simulation.mask` now present the unexpected
  coordinates in the error message.
- :class:`imod.mf6.VerticesDiscretization` now correctly sets the ``xorigins``
  and ``yorigins`` options in the ``.disv`` file. Incorrect origins cause issues
  when splitting models and computing with XT3D on the exchanges.
- :func:`imod.mf6.open_cbc` and :func:`imod.mf6.open_head` now account for
  xorigins and yorigins for models ran with
  :class:`imod.mf6.VerticesDiscretization`. **WARNING**: Given that these were
  set incorrectly in previous versions of iMOD Python (see previous item in this
  list), this means that reading MODFLOW6 DISV output of models generated with a
  previous version of iMOD Python will result in a grid with an erroneous
  offset. You can work around this by creating the model again with this
  version of iMOD Python or newer.
- :meth:`imod.mf6.Modflow6Simulation.split` supports label array with a
  different name than ``"idomain"``.
- :func:`imod.msw.MetaSwapModel.from_imod5_data` now supports the usage of
  relative paths for the extra files block.
- Bug in :meth:`imod.msw.Sprinkling.write` where MetaSWAP svats with surface
  water sprinkling and no groundwater sprinkling activated were not written to
  ``scap_svat.inp``.
- :class:`imod.msw.IdfMapping` swapped order of y_grid and x_grid in dictionary
  for writing the correct order of coordinates in idf_svat.inp.
- Improved performance of :meth:`imod.mf6.Modflow6Simulation.split` and
  :meth:`imod.mf6.Modflow6Simulation.mask` when using dask.
- Fixed bug in :meth:`imod.mf6.Modflow6Simulation.mask` for unstructured grids
  with a spatial dimension that differs from the default ``"mesh2d_nFaces"``.
- Fixed bug in :meth:`imod.mf6.Well.cleanup` and
  :meth:`imod.mf6.LayeredWell.cleanup` which caused an error when called with an
  unstructured discretization.
- Fixed bug in :func:`imod.formats.prj.open_projectfile_data` which caused an
  error when a periods keyword was used having an upper case.
- Poor performance of :meth:`imod.mf6.Well.from_imod5_data` and
  :meth:`imod.mf6.LayeredWell.from_imod5_data` when the ``imod5_data`` contained
  a well system with a large number of wells (>10k).
- :meth:`imod.mf6.River.from_imod5_data`,
  :meth:`imod.mf6.Drainage.from_imod5_data`,
  :meth:`imod.mf6.GeneralHeadBoundary.from_imod5_data` can now deal with
  constant values for variables. One variable per package still needs to be a
  grid.
- Fix bug where an error was thrown in ``get_non_grid_data`` when calling the
  ``.cleanup`` and ``regrid_like`` methods on a boundary condition package with
  a repeated stress. For example, :meth:`imod.mf6.River.cleanup` or
  :meth:`imod.mf6.River.regrid_like`.
- Fix bug where an error was thrown in :class:`imod.mf6.Well` when an entry had
  to be filtered and its ``id`` didn't match the index.
- Improved performance of :class:`imod.mf6.Modflow6Simulation.split` for
  structured models, as unnecessary masking is avoided.
- Fixed warning thrown by type dispatcher about ``~GeoDataFrameType``
- Fixed bug where variables in a package with only a ``"layer"`` coordinate
  could not be regridded or masked.

Changed
~~~~~~~

- :meth:`imod.wq.SeawatModel.write` now throws an error if trying to write in a
  directory with a space in the path. (iMOD-WQ does not support this.)
- `imod.mf6.multimodel.partition_generator.get_label_array` moved to
  :func:`imod.prepare.create_partition_labels`.
- :func:`imod.prepare.create_partition_labels` structured grids are now
  partioned by METIS instead (just like already was the case for unstructured
  grids). This results in more balanced partitions for grids with non-square
  domains or lots of inactive cells. Downside is that the partitions are more
  often than not perfectly rectangular in shape.
- :func:`imod.prepare.create_partition_labels` now returns a griddata with the
  name ``"label"`` instead of ``"idomain"``.
- Upon providing the wrong type to one of the options of
  :class:`imod.mf6.GroundwaterFlowModel`,
  :class:`imod.mf6.GroundwaterTransportModel`, this will throw a
  ``ValidationError`` upon initialization and writing.
- You can now also provide ``repeat_stress`` as dictionary to imod.mf6
  boundary conditions, such as :class:`imod.mf6.River`, :class:`imod.mf6.Drainage`, and
  :class:`imod.mf6.GeneralHeadBoundary`.
- :meth:`imod.mf6.ConstantHead.from_imod5_data`,
  :meth:`imod.mf6.GeneralHeadBoundary.from_imod5_data`,
  :meth:`imod.mf6.River.from_imod5_data`,
  :meth:`imod.mf6.Recharge.from_imod5_data`, and
  :meth:`imod.mf6.Drainage.from_imod5_data` now forward fill data over time,
  instead of clipping, when selecting a start time that is inbetween two data
  records.
- :meth:`imod.mf6.ConstantHead.from_imod5_data` and
  :meth:`imod.mf6.Recharge.from_imod5_data` got extra arguments for
  ``period_data``, ``time_min`` and ``time_max``.
- :func:`imod.prepare.read_imod_legend` now also returns the labels as an extra
  argument. Update your code by changing 
  ``colors, levels = read_imod_legend(...)`` to 
  ``colors, levels, labels = read_imod_legend(...)``.


[1.0.0rc3] - 2025-04-17
-----------------------

Added
~~~~~

- :meth:`imod.msw.MetaSwapModel.clip_box` to clip MetaSWAP models.
- Methods of class :class:`imod.mf6.Modflow6Simulation` can now be logged.
- :func:`imod.prepare.cleanup.cleanup_layered_wel` to clean up wells assigned
  to layers.


Fixed
~~~~~

- Fixed bug where :meth:`imod.mf6.River.clip_box`,
  :meth:`imod.mf6.Drainage.clip_box`, and
  :meth:`imod.mf6.GeneralHeadBoundary.clip_box` threw an error when
  ``time_start`` or ``time_end`` were set to ``None`` and a ``"repeat_stress"``
  was included in the dataset.
- Fixed bug where :meth:`imod.mf6.package.copy` threw an error.
- Sorting issue in :func:`imod.prepare.assign_wells`. This could cause
  :class:`imod.mf6.Well` to assign wells to the wrong cells.
- Fixed crash upon calling :meth:`imod.mf6.Well.clip_box` when the top/bottom
  arguments are specified. This could cause :class:`imod.mf6.Well` to crash
  when wells are located outside the extent of the layer model.


[1.0.0rc2] - 2025-03-05
-----------------------

From this release on, we recommend using `xugrid's regridding utilities
<https://deltares.github.io/xugrid/examples/regridder_overview.html>`_ for
regridding individual grids instead of :class:`imod.prepare.Regridder`. Xugrid's
regridders are tested to be about 10 times faster than
:class:`imod.prepare.Regridder`. There is one small difference: xugrid's
``xugrid.BaryCentricInterpolator`` considers sample points of the destination
grid that lie on the source grid's cell edges to be inside, whereas
:class:`imod.prepare.Regridder` considers them to be outside. This difference is
negligible for most applications, but might create slightly fewer ``np.nan``
values than before.

Removed
~~~~~~~
- ``imod.flow`` module has been removed for generating iMODFLOW models. Use
  ``imod.mf6`` instead to generate MODFLOW 6 models.

Added
~~~~~

- Support for Python 3.13.
- :meth:`imod.mf6.Recharge.from_imod5_data`,
  :meth:`imod.mf6.River.from_imod5_data`,
  :meth:`imod.mf6.Drainage.from_imod5_data`, and
  :meth:`imod.mf6.GeneralHeadBoundary.from_imod5_data` now assign negative layer
  numbers to the first active layer.
- :func:`imod.prepare.DISTRIBUTING_OPTION` got a new setting
  ``by_corrected_thickness``. This matches DISTRCOND=-1 in iMOD5.
- :func:`imod.prepare.cleanup.cleanup_hfb` to clean up HFB geometries.
- :meth:`imod.mf6.HorizontalFlowBarrierResistance.cleanup`,
  :meth:`imod.mf6.SingleLayerHorizontalFlowBarrierResistance.cleanup`,
  to clean up HFB geometries crossing inactive model cells.
- :class:`imod.util.RegridderWeightsCache` to store regridder weights for
  regridding multiple times.
- :class:`imod.util.RegridderType` to specify regridder types.

Changed
~~~~~~~

- :func:`imod.formats.prj.open_projectfile_data` now also assigns negative and
  zero layer numbers to grid coordinates.
- In :class:`imod.mf6.StructuredDiscretization`, IDOMAIN can now respectively be
  > 0 to indicate an active cell and <0 to indicate a vertical passthrough cell,
  consistent with MODFLOW 6. Previously this could only be indicated with 1 and
  -1.
- :meth:`imod.mf6.Well.from_imod5_data` and
  :meth:`imod.mf6.LayeredWell.from_imod5_data` now also accept the argument
  ``times = "steady-state"``, for the simulation is assumed to be "steady-state"
  and well timeseries are averaged.
- The ``drn`` attribute of :class:`imod.prepare.SimulationAllocationOptions` has
  the ``at_elevation`` of :func:`imod.prepare.ALLOCATION_OPTION` option now set
  as default. This means by default drainage cells are placed differently in
  :meth:`imod.mf6.Simulation.from_imod5_data`.
- :class:`imod.mf6.Well`, :class:`imod.mf6.LayeredWell`,
  :func:`imod.prepare.wells.assign_wells`, :meth:`imod.mf6.Well.from_imod5_data`
  and :meth:`imod.mf6.LayeredWell.from_imod5_data` now have default values for
  ``minimum_thickness`` and ``minimum_k`` set to 0.0.
- When intitating a MODFLOW 6 package with a ``layer`` coordinate with
  values <= 0, iMOD Python will throw an error.
- :class:`imod.mf6.HorizontalFlowBarrierResistance`,
  :class:`imod.mf6.HorizontalFlowBarrierSingleLayerResistance` and other HFB now
  validate whether proper type of geometry is provided, respectively Polygon for
  :class:`imod.mf6.HorizontalFlowBarrierResistance`, and LineString for
  :class:`imod.mf6.HorizontalFlowBarrierSingleLayerResistance`.
- Relaxed validation for :class:`imod.msw.MetaSwapModel` if ``FileCopier``
  package is present.
- Change aterisk to dash and tabs to four spaces in ``ValidationError`` messages.
- :func:`imod.prepare.laplace_interpolate` has been simplified, using
  ``scipy.sparse.linalg.cg`` as the backend. We've remove the support for the
  ``ibound`` argument, the ``iter1`` argument has been dropped, ``mxiter`` has
  been renamed to ``maxiter``, ``close`` has been renamed to ``rtol``.
- Moved ``imod.mf6.utilities.regrid.RegridderWeightsCache`` to the
  :class:`imod.util.regrid.RegridderWeightsCache`.

Fixed
~~~~~

- :meth:`imod.mf6.Model.mask_all_packages` now preserves the ``dx`` and
  ``dy`` coordinates
- :meth:`imod.mf6.Well.from_imod5_data` and
  :meth:`imod.mf6.LayeredWell.from_imod5_data` ignore well rates preceding first
  element of ``times``.
- :meth:`imod.mf6.Well.from_imod5_data` and
  :meth:`imod.mf6.LayeredWell.from_imod5_data` now sum the rates of well entries
  that are on the exact same location (same x, y, and depth) instead of taking
  the values of the first entry.
- :meth:`imod.mf6.River.from_imod5_data` now preserves the drainage cells
  created with the ``stage_to_riv_bot_drn_above`` option of
  :func:`imod.prepare.ALLOCATION_OPTION`.
- Bug in :func:`imod.prepare.distribute_riv_conductance` where conductances were
  set to ``np.nan`` for cells where ``stage`` equals ``bottom_elevation`` when
  :func:`imod.prepare.DISTRIBUTING_OPTION` was set to ``by_crosscut_thickness``,
  ``by_crosscut_transmissivity``, ``by_corrected_transmissivity``.
- :meth:`imod.mf6.NodePropertyFlow.from_imod5_data` now defaults to 90 degrees
  for missing layers ``imod5_data`` instead of 0 degrees.
- Bug in :meth:`imod.mf6.Modflow6Simulation.from_imod5_data` where an error was
  raised in case the ``"cap"`` package was present in the ``imod5_data``.
- Bug where :meth:`imod.mf6.LayeredWell.from_imod5_cap_data` and
  :meth:`imod.mf6.Recharge.from_imod5_cap_data` threw an error if the ``"cap"``
  in the ``imod5_data`` had a ``"layer"`` dimension and coordinate.
- :meth:`imod.mf6.LayeredWell.from_imod5_cap_data` will convert the
  ``max_abstraction_groundwater`` and ``max_abstraction_surfacewater`` capacity
  from mm/d to m3/d.
- :class:`imod.msw.TimeOutputControl` now starts counting at 0.0 instead of 1.0,
  like MetaSWAP expects.
- Models imported with :meth:`imod.msw.MetaSwapModel.from_imod5_data` can be
  written with ``validate`` set to True.
- :meth:`imod.mf6.Recharge.from_imod5_cap_data` now returns a 2D array with a
  ``"layer"`` coordinate of ``1`` as otherwise ``primod`` throws an error when
  trying to derive recharge-svat mappings.
- Fixed part of the code that made Pandas, Geopandas, and xarray throw a lot of
  ``FutureWarning`` and ``DeprecationWarning``.
- Fixed performance issue when converting very large wells (>10k) with
  :meth:`imod.mf6.Well.to_mf6_pkg` and :meth:`imod.mf6.LayeredWell.to_mf6_pkg`,
  such as those created with :meth:`imod.mf6.LayeredWell.from_imod5_cap_data`
  for a large grid.
- Fixed issue where an error was thrown when deriving couplings for
  :class:`imod.msw.CouplerMapping` and computing svats in
  :class:`imod.msw.GridData` with ``dask>=2025.2.0``.
- Fixed a bug where :func:`imod.mf6.out.open_cbc` did not properly sum fluxes
  for a single boundary condition package when multiple entries were present in
  the same cell. This never happened with models generated by iMOD Python, as it
  cannot generate these boundary conditions, but could be a problem with models
  generated by iMOD5 and Flopy.
- Removed duplicate entries in ``mod2svat.inp`` generated by
  :class:`imod.msw.CouplerMapping` as MetaSWAP cannot handle this.


[1.0.0rc1] - 2024-12-20
-----------------------

Small post-release fix for installation instructions in documentation.

[1.0.0rc0] - 2024-12-20
-----------------------

Added
~~~~~

- :class:`imod.msw.MeteoGridCopy` to copy existing `mete_grid.inp` files, so
  ASCII grids in large existing meteo databases do not have to be read.
- :class:`imod.msw.CopyFiles` to copy settings and lookup tables in existing
  ``.inp`` files.
- :meth:`imod.mf6.LayeredWell.from_imod5_cap_data` to construct a
  :class:`imod.mf6.LayeredWell` package from iMOD5 data in the CAP package (for
  MetaSWAP). Currently only griddata (IDF) is supported.
- :meth:`imod.mf6.Recharge.from_imod5_cap_data` to construct a recharge package
  for coupling a MODFLOW 6 model to MetaSWAP.
- :meth:`imod.msw.MetaSwapModel.from_imod5_data` to construct a MetaSWAP model
  from data in an iMOD5 projectfile.
- :meth:`imod.msw.MetaSwapModel.write` has a ``validate`` argument, which can be
  used to turn off validation upon writing, use at your own risk!
- :class:`imod.msw.MetaSwapModel` got ``settings`` argument to set simulation
  settings.
- :func:`imod.data.tutorial_03` to load data for the iMOD Documentation
  tutorial.
- :meth:`imod.mf6.Modflow6Simulation.dump` now saves iMOD Python version number.

Fixed
~~~~~

- Fixed bug where :class:`imod.mf6.HorizontalFlowBarrierResistance`,
  :class:`imod.mf6.HorizontalFlowBarrierSingleLayerResistance` and other HFB
  packages could not be allocated to cell edges when idomain in layer 1 was
  largely inactive.
- Fixed bug where :meth:`HorizontalFlowBarrierResistance.clip_box`,
  :meth:`HorizontalFlowBarrierSingleLayerResistance.clip_box` methods only
  returned deepcopy instead of actually clipping the line geometries.
- Fixed bug where :class:`imod.mf6.HorizontalFlowBarrierResistance`,
  :class:`imod.mf6.HorizontalFlowBarrierSingleLayerResistance` and other HFB
  packages could not be clipped or copied with xarray >= 2024.10.0.
- Fixed crash upon calling :meth:`imod.mf6.GroundwaterFlowModel.dump`, when a
  :class:`imod.mf6.HorizontalFlowBarrierResistance`,
  :class:`imod.mf6.HorizontalFlowBarrierSingleLayerResistance` or other HFB
  package was assigned to the model.
- :meth:`imod.mf6.Modflow6Simulation.regrid_like` can now regrid a structured
  model to an unstructured grid.
- :meth:`imod.mf6.Modflow6Simulation.regrid_like` throws a
  ``NotImplementedError`` when attempting to regrid an unstructured model to a
  structured grid.
- :class:`imod.msw.Sprinkling` now correctly writes source svats to
  scap_svat.inp file.
- :func:`imod.evaluate.calculate_gxg`, upon providing a head dataarray chunked
  over time, will no longer error with ``ValueError: Object has inconsistent
  chunks along dimension bimonth. This can be fixed by calling unify_chunks().``
- Improved performance of regridding package data.


Changed
~~~~~~~

- :class:`imod.msw.Infiltration`'s variables ``upward_resistance`` and
  ``downward_resistance`` now require a ``subunit`` coordinate.
- Variables ``max_abstraction_groundwater`` and ``max_abstraction_surfacewater``
  in :class:`imod.msw.Sprinkling` now needs to have a subunit coordinate.
- If ``"cap"`` package present in ``imod5_data``,
  :meth:`imod.mf6.GroundwaterFlowModel.from_imod5_data` now automatically adds a
  well for metaswap sprinkling named ``"msw-sprinkling"``
- Less strict validation for :class:`imod.mf6.HorizontalFlowBarrierResistance`,
  :class:`imod.mf6.HorizontalFlowBarrierSingleLayerResistance` and other HFB packages for
  simulations which are imported with
  :meth:`imod.mf6.Modflow6Simulation.from_imod5_data`
- DeprecationWarning thrown upon initializing :class:`imod.prepare.Regridder`.
  We plan to remove this object in the final 1.0 release. `Use the xugrid
  regridder to regrid individual grids instead.
  <https://deltares.github.io/xugrid/examples/regridder_overview.html>`_ To
  regrid entire MODFLOW 6 packages or simulations, `see the user guide here.
  <https://deltares.github.io/imod-python/user-guide/08-regridding.html>`_.

[0.18.1] - 2024-11-20
---------------------

Added
~~~~~

- :class:`imod.prepare.SimulationAllocationOptions`,
  :class:`imod.prepare.SimulationDistributingOptions`, which are used to store
  default allocation and distributing options respectively.

Fixed
~~~~~

- Relaxed validation for `imod.mf6.StructuredDiscretization` to also support
  cells with zero thickness where IDOMAIN = 0. Before, only cells with a zero
  thickness and IDOMAIN = -1 were supported, else the software threw a ``not all
  values comply with criterion: > bottom``.
- Fix bug where no ``ValidationError`` was thrown if there is an active RCH, DRN,
  GHB, or RIV cell where idomain = -1.

Changed
~~~~~~~

- In :meth:`imod.mf6.Modflow6Simulation.from_imod5_data`, and
  :meth:`imod.mf6.GroundwaterFlowModel.from_imod5_data` the arguments
  ``allocation_options``, ``distributing_options`` are now optional.
- The order of arguments of :meth:`imod.mf6.Modflow6Simulation.from_imod5_data`,
  and :meth:`imod.mf6.GroundwaterFlowModel.from_imod5_data`. It now is
  ``imod5_data, period_data, times, allocation_options, distributing_options, regridder_types``
  instead of:
  ``imod5_data, period_data, allocation_options, distributing_options, times, regridder_types``


[0.18.0] - 2024-11-11
---------------------

Fixed
~~~~~

- Multiple ``HorizontalFlowBarrier`` objects attached to
  :class:`imod.mf6.GroundwaterFlowModel` are merged into a single horizontal
  flow barrier for MODFLOW 6.
- Bug where error would be thrown when barriers in a ``HorizontalFlowBarrier``
  would be snapped to the same cell edge. These are now summed.
- Improve performance validation upon Package initialization
- Improve performance writing ``HorizontalFlowBarrier`` objects
- :func:`imod.mf6.open_cbc` failing with ``flowja=False`` on budget output for
  DISV models if the model contained inactive cells.
- :func:`imod.mf6.open_cbc` now works for 2D and 1D models.
- :func:`imod.prepare.fill` previously assigned to the result of an xarray
  ``.sel`` operation. This might not work for dask backed data and has been
  addressed.
- Added :func:`imod.mf6.open_dvs` to read dependent variable output files like
  the water content file of :class:`imod.mf6.UnsaturatedZoneFlow`.
- `imod.prj.open_projectfile_data` is now able to also read IPF data for
  sprinkling wells in the CAP package.
- Fix that caused iMOD Python to break upon import with numpy >=1.23, <2.0 .
- ValidationError message now contains a suggestion to use the cleanup method,
  if available in the erroneous package.
- Bug where error was thrown when :class:`imod.mf6.NodePropertyFlow` was
  assigned to :class:`imod.mf6.GroundwaterFlowModel` with key different from
  ``"npf"`` upon writing, along with well or horizontal flow barrier packages.


Changed
~~~~~~~

- :class:`imod.mf6.Well` now also validates that well filter top is above well
  filter bottom
- :func:`imod.formats.prj.open_projectfile_data` now also imports well filter
  top and bottom.
- :class:`imod.mf6.Well` now logs a warning if any wells are removed during writing.
- :class:`imod.mf6.HorizontalFlowBarrierResistance`,
  :class:`imod.mf6.HorizontalFlowBarrierMultiplier`,
  :class:`imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic` now uses
  vertical Polygons instead of Linestrings as geometry, and ``"ztop"`` and
  ``"zbottom"`` variables are not used anymore. See
  :func:`imod.prepare.linestring_to_square_zpolygons` and
  :func:`imod.prepare.linestring_to_trapezoid_zpolygons` to generate these
  polygons.
- :func:`imod.formats.prj.open_projectfile_data` now returns well data grouped
  by ipf name, instead of generic, separate number per entry.
- :class:`imod.mf6.Well` now supports wells which have a filter with zero
  length, where ``"screen_top"`` equals ``"screen_bottom"``.
- :class:`imod.mf6.Well` shares the same default ``minimum_thickness`` as
  :func:`imod.prepare.assign_wells`, which is 0.05, before this was 1.0.
- :func:`imod.prepare.allocate_drn_cells`,
  :func:`imod.prepare.allocate_ghb_cells`,
  :func:`imod.prepare.allocate_riv_cells`, now allocate to the first model layer
  when elevations are above or equal to model top for all methods in
  :func:`imod.prepare.ALLOCATION_OPTION`.
- :meth:`imod.mf6.Well.to_mf6_pkg` got a new argument:
  ``strict_well_validation``, which controls the behavior for when wells are
  removed entirely during their assignment to layers. This replaces the
  ``is_partitioned`` argument.
- :func:`imod.prepare.fill` now takes a ``dims`` argument instead of ``by``,
  and will fill over N dimensions. Secondly, the function no longer takes
  an ``invalid`` argument, but instead always treats NaNs as missing.
- Reverted the need for providing WriteContext objects to MODFLOW 6 Model and
  Package objects' ``write`` method. These now use similar arguments to the
  :meth:`imod.mf6.Modflow6Simulation.write` method.
- :class:`imod.msw.CouplingMapping`, :class:`imod.msw.Sprinkling`,
  `imod.msw.Sprinkling.MetaSwapModel`, now take the
  :class:`imod.mf6.mf6_wel_adapter.Mf6Wel` and the
  :class:`imod.mf6.StructuredDiscretization` packages as arguments at their
  respective ``write`` method, instead of upon initializing these MetaSWAP
  objects.
- :class:`imod.msw.CouplingMapping` and :class:`imod.msw.Sprinkling` now take
  the :class:`imod.mf6.mf6_wel_adapter.Mf6Wel` as well argument instead of the
  deprecated ``imod.mf6.WellDisStructured``.


Added
~~~~~

- :meth:`imod.mf6.Modflow6Simulation.from_imod5_data` to import imod5 data
  loaded with :func:`imod.formats.prj.open_projectfile_data` as a MODFLOW 6
  simulation.
- :func:`imod.prepare.linestring_to_square_zpolygons` and
  :func:`imod.prepare.linestring_to_trapezoid_zpolygons` to generate vertical
  polygons that can be used to specify horizontal flow barriers, specifically:
  :class:`imod.mf6.HorizontalFlowBarrierResistance`,
  :class:`imod.mf6.HorizontalFlowBarrierMultiplier`,
  :class:`imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic`.
- :class:`imod.mf6.LayeredWell` to specify wells directly to layers instead
  assigning them with filter depths.
- :func:`imod.prepare.cleanup_drn`, :func:`imod.prepare.cleanup_ghb`,
  :func:`imod.prepare.cleanup_riv`, :func:`imod.prepare.cleanup_wel`. These are
  utility functions to clean up drainage, general head boundaries, and rivers,
  respectively.
- :meth:`imod.mf6.Drainage.cleanup`,
  :meth:`imod.mf6.GeneralHeadboundary.cleanup`, :meth:`imod.mf6.River.cleanup`,
  :meth:`imod.mf6.Well.cleanup` convenience methods to call the corresponding
  cleanup utility functions with the appropriate arguments.
- :meth:`imod.msw.MetaSwapModel.regrid_like` to regrid MetaSWAP models. This is
  still experimental functionality, regridding the :class:`imod.msw.Sprinkling`
  is not yet supported.
- The context :func:`imod.util.context.print_if_error` to print an error instead
  of raising it in a ``with`` statement. This is useful for code snippets which
  definitely will fail.
- :meth:`imod.msw.MetaSwapModel.regrid_like` to regrid MetaSWAP models.
- :meth:`imod.mf6.GroundwaterFlowModel.prepare_wel_for_mf6` to prepare wells for
  MODFLOW 6, for debugging purposes.

Removed
~~~~~~~

- :func:`imod.formats.prj.convert_to_disv` has been removed. This functionality
  has been replaced by :meth:`imod.mf6.Modflow6Simulation.from_imod5_data`. To
  convert a structured simulation to an unstructured simulation, call:
  :meth:`imod.mf6.Modflow6Simulation.regrid_like`


[0.17.2] - 2024-09-17
---------------------

Fixed
~~~~~
- :func:`imod.formats.prj.open_projectfile_data` now reports the path to a
  faulty IPF or IDF file in the error message.
- Support for Numpy 2.0

Added
~~~~~
- Added objects with regrid settings. These can be used to provide custom
  settings: :class:`imod.mf6.regrid.ConstantHeadRegridMethod`,
  :class:`imod.mf6.regrid.DiscretizationRegridMethod`,
  :class:`imod.mf6.regrid.DispersionRegridMethod`,
  :class:`imod.mf6.regrid.DrainageRegridMethod`,
  :class:`imod.mf6.regrid.EmptyRegridMethod`,
  :class:`imod.mf6.regrid.EvapotranspirationRegridMethod`,
  :class:`imod.mf6.regrid.GeneralHeadBoundaryRegridMethod`,
  :class:`imod.mf6.regrid.InitialConditionsRegridMethod`,
  :class:`imod.mf6.regrid.MobileStorageTransferRegridMethod`,
  :class:`imod.mf6.regrid.NodePropertyFlowRegridMethod`,
  :class:`imod.mf6.regrid.RechargeRegridMethod`,
  :class:`imod.mf6.regrid.RiverRegridMethod`,
  :class:`imod.mf6.regrid.SpecificStorageRegridMethod`,
  :class:`imod.mf6.regrid.StorageCoefficientRegridMethod`.

Changed
~~~~~~~
- Instead of providing a dictionary with settings to ``Package.regrid_like``,
  provide one of the following ``RegridMethod`` objects:
  :class:`imod.mf6.regrid.ConstantHeadRegridMethod`,
  :class:`imod.mf6.regrid.DiscretizationRegridMethod`,
  :class:`imod.mf6.regrid.DispersionRegridMethod`,
  :class:`imod.mf6.regrid.DrainageRegridMethod`,
  :class:`imod.mf6.regrid.EmptyRegridMethod`,
  :class:`imod.mf6.regrid.EvapotranspirationRegridMethod`,
  :class:`imod.mf6.regrid.GeneralHeadBoundaryRegridMethod`,
  :class:`imod.mf6.regrid.InitialConditionsRegridMethod`,
  :class:`imod.mf6.regrid.MobileStorageTransferRegridMethod`,
  :class:`imod.mf6.regrid.NodePropertyFlowRegridMethod`,
  :class:`imod.mf6.regrid.RechargeRegridMethod`,
  :class:`imod.mf6.regrid.RiverRegridMethod`,
  :class:`imod.mf6.regrid.SpecificStorageRegridMethod`,
  :class:`imod.mf6.regrid.StorageCoefficientRegridMethod`.
- Renamed ``imod.mf6.LayeredHorizontalFlowBarrier`` classes to
  :class:`imod.mf6.SingleLayerHorizontalFlowBarrierResistance`,
  :class:`imod.mf6.SingleLayerHorizontalFlowBarrierHydraulicCharacteristic`,
  :class:`imod.mf6.SingleLayerHorizontalFlowBarrierMultiplier`,

Fixed
~~~~~
- :func:`imod.formats.prj.open_projectfile_data` now reports the path to a
  faulty IPF or IDF file in the error message.




[0.17.1] - 2024-05-16
---------------------

Added
~~~~~
- Added function :func:`imod.util.spatial.gdal_compliant_grid` to make spatial
  coordinates of a NetCDF interpretable for GDAL (and so QGIS).
- Added ``crs`` argument to :func:`imod.util.spatial.mdal_compliant_ugrid2d`,
  :meth:`imod.mf6.Simulation.dump`, :meth:`imod.mf6.GroundwaterFlowModel.dump`,
  :meth:`imod.mf6.GroundwaterTransportModel.dump`, to add a coordinate reference
  system to dumped files, to ease loading them in QGIS.

Changed
~~~~~~~
- :meth:`imod.mf6.Simulation.dump`, :meth:`imod.mf6.GroundwaterFlowModel.dump`,
  :meth:`imod.mf6.GroundwaterTransportModel.dump` write with necessary
  attributes to NetCDF to make these files interpretable for GDAL (and so QGIS).

Fixed
~~~~~
- Fix missing API docs for ``dump`` and ``write`` methods.


[0.17.0] - 2024-05-13
---------------------

Added
~~~~~
- Added functions to allocate planar grids over layers for the topsystem in
  :func:`imod.prepare.allocate_drn_cells`,
  :func:`imod.prepare.allocate_ghb_cells`,
  :func:`imod.prepare.allocate_rch_cells`,
  :func:`imod.prepare.allocate_riv_cells`, for this multiple options can be
  selected, available in :func:`imod.prepare.ALLOCATION_OPTION`.
- Added functions to distribute conductances of planar grids over layers for the
  topsystem in :func:`imod.prepare.distribute_riv_conductance`,
  :func:`imod.prepare.distribute_drn_conductance`,
  :func:`imod.prepare.distribute_ghb_conductance`, for this multiple options can
  be selected, available in :func:`imod.prepare.DISTRIBUTING_OPTION`.
- :func:`imod.prepare.celltable` supports an optional ``dtype`` argument. This
  can be used, for example, to create celltables of float values.
- Added ``fixed_cell`` option to :class:`imod.mf6.Recharge`. This option is
  relevant for phreatic models, not using the Newton formulation and model cells
  can become inactive. The prefered method for phreatic models is to use the
  Newton formulation, where cells remain active, and this option irrelevant.
- Added support for ``ats_outer_maximum_fraction`` in :class:`imod.mf6.Solution`.
- Added validation for ``linear_acceleration``, ``rclose_option``,
  ``scaling_method``, ``reordering_method``, ``print_option`` and ``no_ptc``
  entries in :class:`imod.mf6.Solution`.

Fixed
~~~~~
- No ``ValidationError`` thrown anymore in :class:`imod.mf6.River` when
  ``bottom_elevation`` equals ``bottom`` in the model discretization.
- When wells outside of the domain are added, an exception is raised with an
  error message stating a well is outside of the domain.
- When importing data from a .prj file, the multipliers and additions specified for
  ipf and idf files are now applied
- Fix bug where y-coords were flipped in :class:`imod.msw.MeteoMapping`

Changed
~~~~~~~
- Replaced csv_output by outer_csvfile and inner_csvfile in
  :class:`imod.mf6.Solution` to match newer MODFLOW 6 releases.
- Changed no_ptc from a bool to an option string in :class:`imod.mf6.Solution`.
- Removed constructor arguments `source` and `target` from
  ``imod.mf6.utilities.regrid.RegridderWeightsCache``, as they were not
  used.
- :func:`imod.mf6.open_cbc` now returns arrays which contain np.nan for cells where
  budget variables are not defined. Based on new budget output a disquisition between
  active cells but zero flow and inactive cells can be made.
- :func:`imod.mf6.open_cbc` now returns package type in return budget names. New format
  is "package type"-"optional package variable"_"package name". E.g. a River package
  named ``primary-sys`` will get a budget name ``riv_primary-sys``. An UZF package
  with name ``uzf-sys1`` will get a budget name ``uzf-gwrch_uzf-sys1`` for the
  groundwater recharge budget from the UZF-CBC.


[0.16.0] - 2024-03-29
---------------------

Added
~~~~~
- The :func:`imod.mf6.model.mask_all_packages` now also masks the idomain array
  of the model discretization, and can be used with a mask array without a layer
  dimension, to mask all layers the same way
- Validation for incompatible settings in the :class:`imod.mf6.NodePropertyFlow`
  and :class:`imod.mf6.Dispersion` packages.
- Checks that only one flow model is present in a simulation when calling
  :func:`imod.mf6.Modflow6Simulation.regrid_like`,
  :func:`imod.mf6.Modflow6Simulation.clip_box` or
  :func:`imod.mf6.Modflow6Simulation.split`
- Added support for coupling a GroundwaterFlowModel and Transport Model i.c.w.
  the 6.4.3 release of MODFLOW. Using an older version of iMOD Python with this
  version of MODFLOW will result in an error.
- :meth:`imod.mf6.Modflow6Simulation.split` supports splitting transport models,
  including multi-species simulations.
- :meth:`imod.mf6.Modflow6Simulation.open_concentration` and
  :meth:`imod.mf6.Modflow6Simulation.open_transport_budget` support opening
  split multi-species simulations.
  :meth:`imod.mf6.Modflow6Simulation.regrid_like` can now regrid simulations
  that have 1 or more transport models.
- added logging to various initialization methods, write methods and dump
  methods. `See the documentation
  <https://deltares.github.io/imod-python/api/generated/logging/imod.logging.html>`_
  how to activate logging.
- added :func:`imod.data.hondsrug_simulation` and
  :func:`imod.data.hondsrug_crosssection` data.
- simulations and models that include a lake package now raise an exception on
  clipping, partitioning or regridding.

Changed
~~~~~~~
- :meth:`imod.mf6.Modflow6Simulation.open_concentration` and
  :meth:`imod.mf6.Modflow6Simulation.open_transport_budget` raise a
  ``ValueError`` if ``species_ls`` is provided with incorrect length.

Fixed
~~~~~
- Incorrect validation error ``data values found at nodata values of idomain``
  for boundary condition packages with a scalar coordinate not set as dimension.
- Fix issue where :func:`imod.idf.open_subdomains` and
  :func:`imod.mf6.Modflow6Simulation.open_head` (for split simulations) would
  return arrays with incorrect ``dx`` and ``dy`` coordinates for equidistant
  data.
- Fix issue where :func:`imod.idf.open_subdomains` returned a flipped ``dy``
  coordinate for nonequidistant data.
- Made :func:`imod.util.round_extent` available again, as it was moved without
  notice. Function now throws a DeprecationWarning to use
  :func:`imod.prepare.spatial.round_extent` instead.
- :meth'`imod.mf6.Modflow6Simulation.write` failed after splitting the
  simulation. This has been fixed.
- modflow options like "print flow", "save flow", and "print input" can now be
  set on :class:`imod.mf6.Well`
- when regridding a :class:`imod.mf6.Modflow6Simulation`,
  :class:`imod.mf6.GroundwaterFlowModel`,
  :class:`imod.mf6.GroundwaterTransportModel` or a :class:`imod.mf6.package`,
  regridding weights are now cached and can be re-used over the different
  objects that are regridded. This improves performance considerably in most use
  cases: when regridding is applied over the same grid cells with the same
  regridder type, but with different values/methods, multiple times.

[0.15.3] - 2024-02-22
---------------------

Fixed
~~~~~
- Add missing required dependencies for installing with ``pip``: loguru and tomli.
- Ensure geopandas and shapely are optional dependencies again when
  installing with ``pip``, and no import errors are thrown.
- Fixed bug where calling ``copy.deepcopy`` on
  :class:`imod.mf6.Modflow6Simulation`, :class:`imod.mf6.GroundwaterFlowModel`
  and :class:`imod.mf6.GroundwaterTransportModel` objects threw an error.


Added
~~~~~
- Developer environment: Added pixi environment ``interactive`` to interactively
  run code. Can be useful to plot data.
- :class:`imod.mf6.ApiPackage` was added. It can be added to both flow and
  transport models, and its presence allows users to interact with libMF6.dll
  through its API.
- Developer environment: Empty python 3.10, 3.11, 3.12 environments where pip
  install and import imod can be tested.



[0.15.2] - 2024-02-16
---------------------

Fixed
~~~~~
- iMOD Python now supports versions of pandas >= 2
- Fixed bugs with clipping :class:`imod.mf6.HorizontalFlowBarrier` for
  structured grids
- Packages and boundary conditions in the ``imod.mf6`` module will now throw an
  error upon initialization if coordinate labels are inconsistent amongst
  variables
- Improved performance for merging structured multimodel MODFLOW 6 output
- Bug where :func:`imod.formats.idf.open_subdomains` did not properly support custom
  patterns
- Added missing validation for ``concentration`` for :class:`imod.mf6.Drainage` and
  :class:`imod.mf6.EvapoTranspiration` package
- Added validation :class:`imod.mf6.Well` package, no ``np.nan`` values are
  allowed
- Fix support for coupling a GroundwaterFlowModel and Transport Model i.c.w.
  the 6.4.3 release of MODFLOW. Using an older version of iMOD Python
  with this version of MODFLOW will result in an error.


Changed
~~~~~~~
- We moved to using `pixi <https://pixi.sh/>`_ to create development
  environments. This replaces the ``imod-environment.yml`` conda environment. We
  advice doing development installations with pixi from now on. `See the
  documentation. <https://deltares.github.io/imod-python/installation.html>`_
  This does not affect users who installed with ``pip install imod``, ``mamba
  install imod`` or ``conda install imod``.
- Changed build system from ``setuptools`` to ``hatchling``. Users who did a
  development install are adviced to run ``pip uninstall imod`` and ``pip
  install -e .`` again. This does not affect users who installed with ``pip
  install imod``, ``mamba install imod`` or ``conda install imod``.
- Decreased lower limit of MetaSWAP validation for x and y limits in the
  ``IdfMapping`` from 0 to -9999999.0.


[0.15.1] - 2023-12-22
---------------------

Fixed
~~~~~
- Made ``specific_yield`` optional argument in
  :class:`imod.mf6.SpecificStorage`, :class:`imod.mf6.StorageCoefficient`.
- Fixed bug where simulations with :class:`imod.mf6.Well` were not partitioned
  into multiple models.
- Fixed erroneous default value for the ``out_of_bounds`` in
  :func:`imod.select.points.point_values`
- Fixed bug where :class:`imod.mf6.Well` could not be assigned to the first cell
  of an unstructured grid.
- HorizontalFlowBarrier package now dropped if completely outside partition in a
  split model.
- HorizontalFlowBarrier package clipped with ``clip_by_grid`` based on active
  cells, consistent with how other packages are treated by this function. This
  affects the :meth:`imod.mf6.HorizontalFlowBarrier.regrid_like` and
  :meth:`imod.mf6.Modflow6Simulation.split` methods.


Changed
~~~~~~~
- All the references to GitLab have been replaced by GitHub references as
  part of the GitHub migration.

Added
~~~~~
- Added comment in Modflow6 exchanges file (GWFGWF) denoting column header.
- Added Python 3.11 support.
- The GWF-GWF exchange options are derived from user created packages (NPF, OC) and
  set automatically.
- Added the ``simulation_start_time`` and ``time_unit`` arguments. To the
  ``Modflow6Simulation.open_`` methods, and ``imod.mf6.out.open_`` functions.
  This converts the ``"time"`` coordinate to datetimes.
- added :meth:`imod.mf6.Modflow6Simulation.mask_all_models` to apply a mask to
  all models under a simulation, provided the simulation is not split and the
  models use the same discretization.


Changed
~~~~~~~
- :meth:`imod.mf6.Well.mask` masks with a 2D grid instead of returning a
  deepcopy of the package.


[0.15.0] - 2023-11-25
---------------------

Fixed
~~~~~
- The Newton option for a :class:`imod.mf6.GroundwaterFlowModel` was being ignored. This has been
  corrected.
- The Contextily packages started throwing errors. This was caused because the
  default tile provider being used was Stamen. However Stamen is no longer free
  which caused Contextily to fail. The default tile provider has been changed to
  OpenStreetMap to resolve this issue.
- :func:`imod.mf6.open_cbc` now reads saved cell saturations and specific discharges.
- :func:`imod.mf6.open_cbc` failed to read unstructured budgets stored
  following IMETH1, most importantly the storage fluxes.
- Fixed support of Python 3.11 by dropping the obsolete ``qgs`` module.
- Bug in :class:`imod.mf6.SourceSinkMixing` where, in case of multiple active
  boundary conditions with assigned concentrations, it would write a ``.ssm``
  file with all sources/sinks on one single row.
- Fixed bug where TypeError was thrown upond calling
  :meth:`imod.mf6.HorizontalFlowBarrier.regrid_like` and
  :meth:`imod.mf6.HorizontalFlowBarrier.mask`.
- Fixed bug where calling :meth:`imod.mf6.Well.clip_box` over only the time
  dimension would remove the index coordinate.
- Validation errors are rendered properly when writing a simulation object or
  regridding a model object.

Changed
~~~~~~~
- The imod-environment.yml file has been split in an imod-environment.yml
  (containing all packages required to run imod-python) and a
  imod-environment-dev.yml file (containing additional packages for developers).
- Changed the way :class:`imod.mf6.Modflow6Simulation`,
  :class:`imod.mf6.GroundwaterFlowModel`,
  :class:`imod.mf6.GroundwaterTransportModel`, and MODFLOW 6 packages are
  represented while printing.
- The grid-agnostic packages :meth:`imod.mf6.Well.regrid_like` and
  :meth:`imod.mf6.HorizontalFlowBarrier.regrid_like` now return a clip with the
  grid exterior of the target grid

Added
~~~~~
- The unit tests results are now published on GitLab
- A ``save_saturation`` option to :class:`imod.mf6.NodePropertyFlow` which saves
  cell saturations for unconfined flow.
- Functions :func:`imod.prepare.layer.get_upper_active_layer_number` and
  :func:`imod.prepare.layer.get_lower_active_layer_number` to return planar
  grids with numbers of the highest and lowest active cells respectively.
- Functions :func:`imod.prepare.layer.get_upper_active_grid_cells` and
  :func:`imod.prepare.layer.get_lower_active_grid_cells` to return boolean
  grids designating respectively the highest and lowest active cells in a grid.
- validation of ``transient`` argument in :class:`imod.mf6.StorageCoefficient`
  and :class:`imod.mf6.SpecificStorage`.
- :meth:`imod.mf6.Modflow6Simulation.open_concentration`,
  :meth:`imod.mf6.Modflow6Simulation.open_head`,
  :meth:`imod.mf6.Modflow6Simulation.open_transport_budget`, and
  :meth:`imod.mf6.Modflow6Simulation.open_flow_budget`, were added as convenience
  methods to open simulation output easier (without having to specify paths).
- The :meth:`imod.mf6.Modflow6Simulation.split` method has been added. This method makes
  it possible for a user to create a Multi-Model simulation. A user needs to
  provide a submodel label array in which they specify to which submodel a cell
  belongs. The method will then create the submodels and split the nested
  packages. The split method will create the gwfgwf exchanges required to
  connect the submodels. At the moment auxiliary variables ``cdist`` and
  ``angldegx`` are only computed for structured grids.
- The label array can be generated through a convenience function
  :func:`imod.mf6.partition_generator.get_label_array`
- Once a split simulation has been executed by MF6, we find head and balance
  results in each of the partition models. These can now be merged into head and
  balance datasets for the original domain using
  :meth:`imod.mf6.Modflow6Simulation.open_concentration`,
  :meth:`imod.mf6.Modflow6Simulation.open_head`,
  :meth:`imod.mf6.Modflow6Simulation.open_transport_budget`,
  :meth:`imod.mf6.Modflow6Simulation.open_flow_budget`.
  In the case of balances, the exchanges through the partition boundary are not
  yet added to this merged balance.
- Settings such as ``save_flows`` can be passed through
  :meth:`imod.mf6.SourceSinkMixing.from_flow_model`
- Added :class:`imod.mf6.LayeredHorizontalFlowBarrierHydraulicCharacteristic`,
  :class:`imod.mf6.LayeredHorizontalFlowBarrierMultiplier`,
  :class:`imod.mf6.LayeredHorizontalFlowBarrierResistance`, for horizontal flow
  barriers with a specified layer number.


Removed
~~~~~~~
- Tox has been removed from the project.
- Dropped support for writing .qgs files directly for QGIS, as this was hard to
  maintain and rarely used. To export your model to QGIS readable files, call
  the ``dump`` method :class:`imod.mf6.Modflow6Simulation` with ``mdal_compliant=True``.
  This writes UGRID NetCDFs which can read as meshes in QGIS.
- Removed ``declxml`` from repository.

[0.14.1] - 2023-09-07
---------------------

Changed
~~~~~~~

- TWRI MODFLOW 6 example uses the grid-agnostic :class:`imod.mf6.Well`
  package instead of the ``imod.mf6.WellDisStructured`` package.

Fixed
~~~~~

- :class:`imod.mf6.HorizontalFlowBarrier` would write to a binary file by
  default. However, the current version of MODFLOW 6 does not support this.
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

- :meth:`imod.mf6.Modflow6Simulation.regrid_like` to regrid a Modflow6 simulation to a
  new grid (structured or unstructured), using `xugrid's regridding
  functionality.
  <https://deltares.github.io/xugrid/examples/regridder_overview.html>`_
  Variables are regridded with pre-selected methods. The regridding
  functionality is useful for a variety of applications, for example to test the
  effect of different grid sizes, to add detail to a simulation (by refining the
  grid) or to speed up a simulation (by coarsening the grid) to name a few
- :meth:`imod.mf6.Package.regrid_like` to regrid packages. The user can
  specify their own custom regridder types and methods for variables.
- :meth:`imod.mf6.Modflow6Simulation.clip_box` got an extra argument
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
- :meth:`imod.mf6.Modflow6Simulation._validate` now print all validation errors for all
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
  MODFLOW 6 output files.
- :func:`imod.util.spatial.from_mdal_compliant_ugrid2d` to "restack" the variables that
  have have been "unstacked" in :func:`imod.util.spatial.mdal_compliant_ugrid2d`.
- Added support for the Modflow6 Lake package
- :func:`imod.select.points_in_bounds`, :func:`imod.select.points_indices`,
  :func:`imod.select.points_values` now support unstructured grids.
- Added support for the MODFLOW 6 Lake package: :class:`imod.mf6.Lake`,
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
  which can be used to provide keyword arguments to the writing of the MODFLOW 6
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
  results in invalid MODFLOW 6 input for 2D grid data, such as DIS top.
- ``imod.flow.ImodflowModel.write`` no longer writes incorrect project
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

- ``imod.flow.ImodflowModel.write`` now supports writing a
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
- ``imod.flow.ImodflowModel.write`` will set the timestring in the
  projectfile to ``steady-state`` for ``BoundaryConditions`` without a time
  dimension.
- Added ``imod.flow.OutputControl`` as this was still missing.
- :func:`imod.ipf.read` will no longer error when an associated files with 0
  rows is read.
- :func:`imod.evaluate.calculate_gxg` now correctly uses (March 14, March
  28, April 14) to calculate GVG rather than (March 28, April 14, April 28).
- :func:`imod.mf6.out.open_cbc` now correctly loads boundary fluxes.
- :meth:`imod.prepare.LayerRegridder.regrid` will now correctly skip values
  if ``top_source`` or ``bottom_source`` are NaN.
- :func:`imod.gen.write` no longer errors on dataframes with empty columns.
- ``imod.mf6.BoundaryCondition.set_repeat_stress`` reinstated. This is
  a temporary measure, it gives a deprecation warning.

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
- ``imod.flow.ImodflowModel.time_discretization``,
  :meth:`imod.wq.SeawatModel.time_discretization`,
  :meth:`imod.mf6.Modflow6Simulation.time_discretization`,
  are renamed to:
  ``imod.flow.ImodflowModel.create_time_discretization``,
  :meth:`imod.wq.SeawatModel.create_time_discretization`,
  :meth:`imod.mf6.Modflow6Simulation.create_time_discretization`,
- Moved tests inside `imod` directory, added an entry point for pytest fixtures.
  Running the tests now requires an editable install, and also existing
  installations have to be reinstalled to run the tests.
- The ``imod.mf6`` model packages now all run type checks on input. This is a
  breaking change for scripts which provide input with an incorrect dtype.
- :class:`imod.mf6.Solution` now requires a `model_names` argument to specify
  which models should be solved in a single numerical solution. This is
  required to simulate groundwater flow and transport as they should be
  in separate solutions.
- When writing MODFLOW 6 input option blocks, a NaN value is now recognized as
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
- ``imod.flow.ImodflowModel.create_time_discretization``,
  :meth:`imod.wq.SeawatModel.create_time_discretization`,
  :meth:`imod.mf6.Modflow6Simulation.create_time_discretization`, now have a
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
   structured (DIS) MODFLOW 6 models. The data is read lazily into xarray
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
-  Added ``imod.flow.ImodflowModel`` to write to model iMODFLOW project
   file.
-  :meth:`imod.mf6.Modflow6Simulation.write` now has a ``binary`` keyword. When set
   to ``False``, all MODFLOW 6 input is written to text rather than binary files.
-  Added :class:`imod.mf6.DiscretizationVertices` to write MODFLOW 6 DISV model
   input.
-  Packages for :class:`imod.mf6.GroundwaterFlowModel` will now accept
   :class:`xugrid.UgridDataArray` objects for (DISV) unstructured grids, next to
   :class:`xarray.DataArray` objects for structured (DIS) grids.
-  Transient wells are now supported in ``imod.mf6.WellDisStructured`` and
   ``imod.mf6.WellDisVertices``.
-  :func:`imod.util.to_ugrid2d` has been added to convert a (structured) xarray
   DataArray or Dataset to a quadrilateral UGRID dataset.
-  Functions created to create empty DataArrays with greater ease:
   :func:`imod.util.empty_2d`, :func:`imod.util.empty_2d_transient`,
   :func:`imod.util.empty_3d`, and :func:`imod.util.empty_3d_transient`.
-  :func:`imod.util.where` has been added for easier if-then-else operations,
   especially for preserving NaN nodata values.
-  :meth:`imod.mf6.Modflow6Simulation.run` has been added to more easily run a model,
   especially in examples and tests.
-  :func:`imod.mf6.open_cbc` and :func:`imod.mf6.open_hds` will automatically
   return a ``xugrid.UgridDataArray`` for MODFLOW 6 DISV model output.

Changed
~~~~~~~

-  Documentation overhaul: different theme, add sample data for examples, add
   Frequently Asked Questions (FAQ) section, restructure API Reference. Examples
   now ru
-  Datetime columns in IPF associated files (via
   :func:`imod.ipf.write_assoc`) will not be placed within quotes, as this can
   break certain iMOD batch functions.
-  :class:`imod.mf6.Well` has been renamed into ``imod.mf6.WellDisStructured``.
-  :meth:`imod.mf6.GroundwaterFlowModel.write` will now write package names
   into the simulation namefile.
-  :func:`imod.mf6.open_cbc` will now return a dictionary with keys
   ``flow-front-face, flow-lower-face, flow-right-face`` for the face flows,
   rather than ``front-face-flow`` for better consistency.
-  Switched to composition from inheritance for all model packages: all model
   packages now contain an internal (xarray) Dataset, rather than inheriting
   from the xarray Dataset.
-  :class:`imod.mf6.SpecificStorage` or :class:`imod.mf6.StorageCoefficient` is
   now mandatory for every MODFLOW 6 model to avoid accidental steady-state
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
-  :func:`imod.util.spatial.coord_reference` now returns a scalar cellsize if coordinate is equidistant.
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
-  Docstrings for the MODFLOW 6 classes in :mod:`imod.mf6`
-  :meth:`imod.select.upper_active_layer` function to get the upper active layer from ibound ``xr.DataArray``

Changed
~~~~~~~

-  ``imod.idf.read`` is deprecated, use :func:`imod.idf.open` instead
-  ``imod.rasterio.read`` is deprecated, use :func:`imod.rasterio.open` instead

Fixed
~~~~~

-  :meth:`imod.prepare.reproject` working instead of silently failing when given a ``"+init=ESPG:XXXX`` CRS string

[0.8.0] - 2019-10-14
--------------------

Added
~~~~~
-  Laplace grid interpolation :meth:`imod.prepare.laplace_interpolate`
-  Experimental MODFLOW 6 structured model write support :mod:`imod.mf6`
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
