iMOD5 backwards compatibility
=============================

iMOD Python tries to be as backwards compatible with iMOD5 as our resources
allow. Below you can find some known issues, notes, and more detailed tables
with the status of support for important iMOD5 features. If you miss an
important feature or run into any issues, `feel free to open an issue on our
issueboard <https://github.com/Deltares/imod-python/issues>`_.

Known issues
------------

- ISG files cannot be read directly. The workaround is to rasterize the ISG
  files to IDF files using the iMOD5 BATCH function ISGGRID.
- MetaSWAP sprinkling wells defined as IPF files are not supported.
- MetaSWAP's "flexible drainage" ("Peilgestuurde drainage" in Dutch) is not
  supported.
- IPEST is not supported.
- *Edge case*: When a well is located on the x-boundary of a cell in the model
  grid, the well is placed by iMOD5 in the western cell, whereas iMOD Python
  places the well in the eastern cell. This is because iMOD Python conforms to
  GDAL in how it looks up points in a grid. The workaround is to move the well
  slightly away from the boundary, by subtracting a small value from the
  x-coordinate of the well (e.g. 0.1 mm).
- There is a bug in iMOD5 in how the HFB's hydraulic characteristic is computed.
  iMDO5 adds the background resistance of the aquifer to the HFB resistance when
  writing the MODFLOW 6 input files. The background resistance is already
  accounted for by MODFLOW 6 internally, so in practice this means the background
  resistance is added twice. This is especially noticeable when a HFB intersects
  a low permeable layer. The workaround in iMOD5 is to specify a negative HFB
  factor in the projectfile, which makes iMOD5 to skip adding the background
  resistance.
- By specifying a negative HFB factor in the projectfile, the iMOD5
  documentation mentions this will turn on the option to override a resistance
  in between cells with the HFB resistance. This is NOT supported by MODFLOW 6,
  as MODFLOW 6 always adds the background resistance to the HFB resistance to
  compute the resistance inbetween cells.


Notes
-----

- if STO package in projectfile is absent, an error is thrown when trying to write
  the :class:`imod.mf6.Modflow6Simulation` object to disk. This is due to the
  fact that the STO package is mandatory in iMOD Python. A workaround is to add a
  :class:`imod.mf6.StorageCoefficient` package to the projectfile before calling
  :meth:`imod.mf6.Modflow6Simulation.write`.
- When importing models with the ANI package, make sure to activate the "XT3D"
  in the NodePropertyFlow package of the model.
- Solver settings (PCG) are NOT imported from iMOD5, instead a solver settings
  preset from MODFLOW 6 ("Moderate") is set. This is because the solvers between
  iMODLFOW and MODFLOW 6 are different. You are advised to test settings
  yourself.
- The imported iMOD5 discretization for the model is created by taking the
  smallest grid and finest resolution amongst the TOP, BOT, and BND grids. This
  differs from iMOD5, where the first BND grid is used as target grid. All input
  grids are regridded towards this target grid. Therefore, be careful when you
  have a very fine resolution in one of these packages.


Files
-----

Here an overview of iMOD5 files supported:

.. csv-table::
   :header-rows: 1

    iMOD5 file,iMOD Python function, Workaround
    Open projectfile data (.prj),":func:`imod.formats.prj.open_projectfile_data`, :func:`imod.formats.prj.read_projectfile`",
    Open runfile data (.run),,Convert to ``.prj`` file with iMOD5 BATCH function RUNFILE
    Open point data (.ipf),:func:`imod.formats.ipf.read`,
    Save point data (.ipf),:func:`imod.formats.ipf.save`,
    Open raster data (.idf),:func:`imod.formats.idf.open`,
    Save raster data (.idf),:func:`imod.formats.idf.save`,
    Open vector data: 2D & 3D (.gen),:func:`imod.formats.gen.read`,
    Save vector data (.gen),:func:`imod.formats.gen.write`,
    Open 1D network data (.isg),,Rasterize to ``.idf`` files with iMOD5 BATCH function ISGGRID
    Open raster data (.asc),:func:`imod.formats.rasterio.open`,
    Open legend file (.leg),:func:`imod.visualize.read_imod_legend`,

MODFLOW 6
---------

Here an overview of iMOD5 MODFLOW 6 features:

.. csv-table::
   :header-rows: 1

    iMOD5 pkg,functionality,iMOD Python function/method
    *Model*,From iMOD5 data,:meth:`imod.mf6.Modflow6Simulation.from_imod5_data`
    *Model*,Regrid,:meth:`imod.mf6.Modflow6Simulation.regrid_like`
    *Model*,Clip,:meth:`imod.mf6.Modflow6Simulation.clip_box`
    *Model*,Validate,:meth:`imod.mf6.Modflow6Simulation.write`
    BND,IBOUND to IDOMAIN,:meth:`imod.mf6.StructuredDiscretization.from_imod5_data`
    "BND, TOP, BOT",Import from grid (IDF),:meth:`imod.mf6.StructuredDiscretization.from_imod5_data`
    "BND, TOP, BOT",Align iMOD5 input grids,:meth:`imod.mf6.StructuredDiscretization.from_imod5_data`
    "BND, TOP, BOT",Regrid,:meth:`imod.mf6.StructuredDiscretization.regrid_like`
    "BND, TOP, BOT",Clip,:meth:`imod.mf6.StructuredDiscretization.clip_box`
    "BND, SHD",set constant heads starting head (IBOUND = -1),:meth:`imod.mf6.ConstantHead.from_imod5_shd_data`
    "BND, CHD",set constant heads (IBOUND = -1),:meth:`imod.mf6.ConstantHead.from_imod5_data`
    "KDW, VCW, KVV, THK",Quasi-3D permeability from grid (IDF),Quasi-3D is only supported by MODFLOW2005. MODFLOW 6 requires fully 3D.
    "KHV, KVA",3D permeability from grid (IDF),:meth:`imod.mf6.NodePropertyFlow.from_imod5_data`
    ANI,Set horizontal anistropy ,:meth:`imod.mf6.NodePropertyFlow.from_imod5_data`
    "KHV, KVA, ANI",Align iMOD5 input grids,:meth:`imod.mf6.NodePropertyFlow.from_imod5_data`
    "KHV, KVA, ANI",Regrid,:meth:`imod.mf6.NodePropertyFlow.regrid_like`
    "KHV, KVA, ANI",Clip,:meth:`imod.mf6.NodePropertyFlow.clip_box`
    "STO, SPY",From grid (IDF),:meth:`imod.mf6.StorageCoefficient.from_imod5_data`
    "STO, SPY",Regrid,:meth:`imod.mf6.StorageCoefficient.regrid_like`
    "STO, SPY",Clip,:meth:`imod.mf6.StorageCoefficient.clip_box`
    RCH,From grid (IDF),:meth:`imod.mf6.Recharge.from_imod5_data`
    RCH,Regrid,:meth:`imod.mf6.Recharge.regrid_like`
    RCH,Clip,:meth:`imod.mf6.Recharge.clip_box`
    CHD,From grid (IDF),:meth:`imod.mf6.ConstantHead.from_imod5_data`
    CHD,Regrid,:meth:`imod.mf6.ConstantHead.regrid_like`
    CHD,Clip,:meth:`imod.mf6.ConstantHead.clip_box`
    GHB,Auto placement (IDEFLAYER),":meth:`imod.mf6.GeneralHeadBoundary.from_imod5_data`, :func:`imod.prepare.allocate_ghb_cells`"
    GHB,Distribute conductances (DISTRCOND),":meth:`imod.mf6.GeneralHeadBoundary.from_imod5_data`, :func:`imod.prepare.distribute_ghb_conductance`"
    GHB,Cleanup,":meth:`imod.mf6.GeneralHeadBoundary.cleanup`, :func:`imod.prepare.cleanup_ghb`"
    GHB,From grid (IDF),:meth:`imod.mf6.GeneralHeadBoundary.from_imod5_data`
    GHB,Align iMOD5 input grids ,:meth:`imod.mf6.GeneralHeadBoundary.from_imod5_data`
    GHB,Regrid,:meth:`imod.mf6.GeneralHeadBoundary.regrid_like`
    GHB,Clip,:meth:`imod.mf6.GeneralHeadBoundary.clip_box`
    DRN,Auto placement (IDEFLAYER),":meth:`imod.mf6.Drainage.from_imod5_data`, :func:`imod.prepare.allocate_drn_cells`"
    DRN,Distribute conductances (DISTRCOND),":meth:`imod.mf6.Drainage.from_imod5_data`, :func:`imod.prepare.distribute_drn_conductance`"
    DRN,Cleanup,":meth:`imod.mf6.Drainage.cleanup`, :func:`imod.prepare.cleanup_drn`"
    DRN,From grid (IDF),:meth:`imod.mf6.Drainage.from_imod5_data`
    DRN,Align iMOD5 input grids ,:meth:`imod.mf6.Drainage.from_imod5_data`
    DRN,Regrid,:meth:`imod.mf6.Drainage.regrid_like`
    DRN,Clip,:meth:`imod.mf6.Drainage.clip_box`
    RIV,Infiltration factors (IFF),":meth:`imod.mf6.River.from_imod5_data`, :meth:`imod.prepare.split_conductance_with_infiltration_factor`"
    RIV,Auto placement (IDEFLAYER),":meth:`imod.mf6.River.from_imod5_data`, :func:`imod.prepare.allocate_riv_cells`"
    RIV,Distribute conductances (DISTRCOND),":meth:`imod.mf6.River.from_imod5_data`, :func:`imod.prepare.distribute_riv_conductance`"
    RIV,Cleanup,":meth:`imod.mf6.River.cleanup`, :func:`imod.prepare.cleanup_riv`"
    RIV,From grid (IDF),:meth:`imod.mf6.River.from_imod5_data`
    RIV,Align iMOD5 input grids ,:meth:`imod.mf6.River.from_imod5_data`
    RIV,Regrid,:meth:`imod.mf6.River.regrid_like`
    RIV,Clip,:meth:`imod.mf6.River.clip_box`
    "ISG, SFT",From 1D network (ISG),
    SFR,From 1D network (ISG),
    HFB,From 2D vector (GEN),:meth:`imod.mf6.SingleLayerHorizontalFlowBarrierResistance.from_imod5_data`
    HFB,From 3D vector (GEN),:meth:`imod.mf6.HorizontalFlowBarrierResistance.from_imod5_data`
    HFB,Snap vector to grid edges,":meth:`imod.mf6.SingleLayerHorizontalFlowBarrierResistance.to_mf6_pkg`, :meth:`imod.mf6.HorizontalFlowBarrierResistance.to_mf6_pkg`"
    HFB,"Auto placement, account for not fully penetrating barriers",:meth:`imod.mf6.HorizontalFlowBarrierResistance.to_mf6_pkg`
    HFB,Clip,":meth:`imod.mf6.SingleLayerHorizontalFlowBarrierResistance.clip_box`, :meth:`imod.mf6.HorizontalFlowBarrierResistance.clip_box`"
    HFB,Cleanup,
    WEL,From point data with timeseries (IPF),":meth:`imod.mf6.LayeredWell.from_imod5_data`, :meth:`imod.mf6.Well.from_imod5_data`"
    WEL,Auto placement,":meth:`imod.mf6.LayeredWell.to_mf6_pkg`, :meth:`imod.mf6.Well.to_mf6_pkg`"
    WEL,Cleanup,":meth:`imod.mf6.Well.cleanup`, :func:`imod.prepare.cleanup_wel`"
    WEL,Clip,":meth:`imod.mf6.LayeredWell.clip_box`, :meth:`imod.mf6.Well.clip_box`"

MetaSWAP
--------

An overview of the support for iMOD5's MetaSWAP features:

.. csv-table::
   :header-rows: 1

    iMOD5 pkg, MetaSWAP file, functionality,iMOD Python function/method
    *Model*,``para_sim.inp``,From grids (IDF),:meth:`imod.msw.MetaSwapModel.from_imod5_data`
    *Model*,,Regrid,:meth:`imod.msw.MetaSwapModel.regrid_like`
    *Model*,,Clip,:meth:`imod.msw.MetaSwapModel.clip_box`
    *Model*,``mod2svat.inp``,Coupling,":meth:`imod.msw.MetaSwapModel.from_imod5_data`, :class:`imod.msw.CouplerMapping`"
    *Model*,``idf_svat.ipn``,IDF output,":meth:`imod.msw.MetaSwapModel.from_imod5_data`, :class:`imod.msw.IdfMapping`"
    CAP,``area_svat.inp``,Grid Data,:meth:`imod.msw.GridData.from_imod5_data`
    CAP,``svat2swnr_roff.inp``,Ponding,:meth:`imod.msw.Ponding.from_imod5_data`
    CAP,``infi_svat.inp``,Infiltration,:meth:`imod.msw.Infiltration.from_imod5_data`
    CAP,``uscl_svat.inp``,Perched Water Table,:meth:`imod.msw.ScalingFactors.from_imod5_data`
    CAP,``uscl_svat.inp``,Scaling factors,:meth:`imod.msw.ScalingFactors.from_imod5_data`
    CAP,,Stage-steered drainage,
    CAP,``mete_grid.inp``,Meteogrids,":meth:`imod.msw.MeteoGridCopy.from_imod5_data`, :meth:`imod.msw.PrecipitationMapping.from_imod5_data`, :meth:`imod.msw.EvapotranspirationMapping.from_imod5_data`"
    CAP,``mete_stat.inp``,Meteostations,
    CAP,``scap_svat.inp``,Sprinkling,:meth:`imod.msw.Sprinkling.from_imod5_data`
    CAP,,Sprinkling wells grid (IDF),:meth:`imod.mf6.LayeredWell.from_imod5_cap_data`
    CAP,,Sprinkling wells points (IPF),
    CAP,,Align iMOD5 input grids,

Postprocessing
--------------

The following post-processing features are supported:

.. csv-table::
   :header-rows: 1

    iMOD5 functionality,iMOD Python function/method
    Open heads,":meth:`imod.mf6.Modflow6Simulation.open_head`, :func:`imod.mf6.open_hds`"
    Open budgets,":meth:`imod.mf6.Modflow6Simulation.open_flow_budget`, :func:`imod.mf6.open_cbc`"
    Compute GXG,:func:`imod.evaluate.calculate_gxg`
    Compute waterbalance,:func:`imod.evaluate.facebudget`

Visualization
-------------

The following visualization features are supported. `For interactively viewing
your data, see our iMOD Viewer
<https://deltares.github.io/iMOD-Documentation/viewer.html>`_. 

.. csv-table::
   :header-rows: 1

    iMOD5 functionality,iMOD Python function/method
    Plot cross-section,:func:`imod.visualize.cross_section`
    Plot map,:func:`imod.visualize.plot_map`
    Quiverplot,:func:`imod.visualize.quiver`
    Streamplot,:func:`imod.visualize.streamfunction`
    Water balance,:func:`imod.visualize.waterbalance_barchart`
    3D plot,:class:`imod.visualize.GridAnimation3D`

iMOD BATCH glossary
-------------------

Here is a glossary of the iMOD5 BATCH functions and their arguments, and which
iMOD Python argument for a function to look for.

RUNFILE
*******

The RUNFILE BATCH function is used in iMOD5 to create a MODFLOW 6 runfile or
namfile from an iMOD5 projectfile. The following table lists the arguments of
the function and a pointer to the equivalent iMOD Python function and argument
of this function.

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python, argument
   PRJFILE_IN, Name of a projectfile that need to be used to create a runfile specified by RUNFILE_OUT or a namfile specified by NAMFILE_OUT e.g. PRJFILE_IN=D:\PRJFILES\MODEL.PRJ., :func:`imod.formats.prj.open_projectfile_data`, ``path``
   NAMFILE_OUT, Name of a nam-file that will be created e.g. NAMFILE_OUT=D:\NAMFILES\MODEL.NAM, :meth:`imod.mf6.Modflow6Simulation.write`, ``directory``
   ISS, Type of time configuration to be added to the RUNFILE or NAMFILE; for transient enter ISS=1 and for steady state enter ISS=0., :class:`imod.mf6.StorageCoefficient`, ``transient``
   SDATE, Starting date of the simulation in yyyymmddhhmmss, :meth:`imod.mf6.Modflow6Simulation.from_imod5_data`, ``times``
   EDATE, End date of the simulation in yyyymmddhhmmss, :meth:`imod.mf6.Modflow6Simulation.from_imod5_data`, ``times``
   ITT, Time interval category, :meth:`imod.mf6.Modflow6Simulation.from_imod5_data`, ``times``
   IDT, Time interval of the time steps corresponding to the chosen time interval category ITT e.g. IDT=7 to denote the 7 days whenever ITT=3, :meth:`imod.mf6.Modflow6Simulation.from_imod5_data`, ``times``
   ISTEADY, ISTEADY=1 to include an initial steady-state time step to the model. This will add packages with the time stamp STEADY-STATE to the first stress-period of your model., :class:`imod.mf6.StorageCoefficient`, ``transient``
   NSTEP, Number time step within each stress period, :class:`imod.mf6.TimeDiscretization`, ``n_timesteps``
   NMULT, Multiplication factor in which the step size of each subsequent time step will increase, :class:`imod.mf6.TimeDiscretization`, ``timestep_multiplier``
   INFFCT, Use this keyword to generate two RIV-elements to compensate for a given infiltration factor, :meth:`imod.mf6.River.from_imod5_data`, 
   IDEFLAYER, Assign river-elements to model layers, :meth:`imod.mf6.River.from_imod5_data`, ``allocation_option``
   DISTRCOND, Distribute conductances over the river-elements, :meth:`imod.mf6.River.from_imod5_data`, ``distributing_option``
   NEWTON, "Use Newton-Raphson formulation for groundwater flow between connected, convertible groundwater cells", :class:`imod.mf6.GroundwaterFlowModel`, ``newton_raphson``
   UNCONFINED, Include unconfined conditions for model layers, :class:`imod.mf6.NodePropertyFlow`, ``icelltype``
   DEFUNCONF, Specify spatially whether the UNCONFINED configuration needs to be applied, :class:`imod.mf6.NodePropertyFlow`, ``icelltype``
   THICKSTRT, Minimal thickness of an aquifer which becomes in active in case the given starting head is below that level, :class:`imod.mf6.NodePropertyFlow`, ``starting_head_as_confined_thickness``
   SPECIFIC-STORAGE, Denote that specific storage is entered in the PRJ file instead of storage coefficients, :class:`imod.mf6.SpecificStorage`, 
   WINDOW, "Specify a window (X1,Y1,X2,Y2) for which the constructed RUNFILE will be clipped", :meth:`imod.mf6.Modflow6Simulation.clip_box`, "``x_min``, ``x_max``, ``y_min``, ``y_max``"
   CELLSIZE, Specify a cell size to be used, :meth:`imod.mf6.Modflow6Simulation.regrid_like`, ``target_grid``
   APPLYCHD, Specify APPLYCHD=1 to insert constant head boundary conditions around the model, :meth:`imod.mf6.Modflow6Simulation.clip_box`, ``states_for_boundary``
   MIXELM, Advection scheme, ":class:`imod.mf6.AdvectionTVD`, :class:`imod.mf6.AdvectionUpstream`, :class:`imod.mf6.AdvectionCentral`", 
   NADVFD, Weighting scheme Finite-difference, ":class:`imod.mf6.AdvectionUpstream`, :class:`imod.mf6.AdvectionCentral`", 
   ISOLVE, Start a simulation after generating a RUNFILE or NAMFILE, :meth:`imod.mf6.Modflow6Simulation.run`, 
   MODFLOW6, MODFLOW 6 executable, :meth:`imod.mf6.Modflow6Simulation.run`, ``mf6path``

Some things previously configurable in iMOD5 are fixed in iMOD Python:

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python
   ICONSISTENCY=1 & MINTHICKNESS=0.0, "Correct layer thickness of ``=< MINTHICKNESS`` automatically. This fixed setting combination is also enforced by iMOD5 for MODFLOW6 models", :meth:`imod.mf6.StructuredDiscretization.from_imod5_data`
   SSYSTEM=0, "Aggregating packages of the same type together is not supported yet in iMOD Python.", 
   ICONCHK=0, "Correct drainage levels automatically during simulation. ICONCHK=0 is also enforced by iMOD5 for MODFLOW6 models and is not supported in iMOD Python.", 
   DWEL=1, "Overrule any intermediate dates specfied for the WEL package in the PRJ file.", :meth:`imod.mf6.Well.from_imod5_data`, 

