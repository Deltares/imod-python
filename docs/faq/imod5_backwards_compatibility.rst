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
    "KHV, KVA, ANI",Regrid,:meth:`NPF.regrid_like <imod.mf6.NodePropertyFlow.regrid_like>`
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
of this function. The method that contains most of the logic of the RUNFILE BATCH
function is :meth:`imod.mf6.Modflow6Simulation.from_imod5_data`.

.. csv-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 5, 20, 20, 20

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

Some settings previously configurable in iMOD5 are fixed in iMOD Python:

.. csv-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 5, 20, 20

   BATCH argument, description, iMOD Python
   ICONSISTENCY=1 & MINTHICKNESS=0.0, "Correct layer thickness of ``=< MINTHICKNESS`` automatically. This fixed setting combination is also enforced by iMOD5 for MODFLOW6 models", :meth:`DIS.from_imod5_data <imod.mf6.StructuredDiscretization.from_imod5_data>`
   SSYSTEM=0, "Aggregating packages of the same type together is not supported yet in iMOD Python.", 
   ICONCHK=0, "Correct drainage levels automatically during simulation. ICONCHK=0 is also enforced by iMOD5 for MODFLOW6 models and is not supported in iMOD Python.", 
   DWEL=1, "Overrule any intermediate dates specfied for the WEL package in the PRJ file.", :meth:`WEL.from_imod5_data <imod.mf6.Well.from_imod5_data>`

GENSNAPTOGRID
*************

The GENSNAPTOGRID function can be used to rasterize a GEN file for a given
raster. See
:meth:`imod.mf6.SingleLayerHorizontalFlowBarrierResistance.snap_to_grid` and
:meth:`imod.mf6.HorizontalFlowBarrierResistance.snap_to_grid` to snap lines to
the grid. The table below lists pointers to the functions and arguments that can
be used to achieve full feature parity with the iMOD5 BATCH function.

.. csv-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 5, 20, 20, 20

   BATCH argument, description, iMOD Python, argument
   GENFILE_IN, Name of a GEN file that needs to be snapped to the grid, ":meth:`HFB.from_imod5_data <imod.mf6.SingleLayerHorizontalFlowBarrierResistance.from_imod5_data>`, :func:`imod.formats.gen.read`", ``path``
   GENFILE_OUT, Name of a GEN file that will be created, :func:`imod.formats.gen.write`, ``path``
   IDFFILE, Name of an IDF file that will be used to snap the GEN file to, :func:`imod.formats.idf.open`, ``path``
   WINDOW, Enter the coordinates of the window that need to be computed solely, :meth:`HFB.snap_to_grid <imod.mf6.SingleLayerHorizontalFlowBarrierResistance.snap_to_grid>`, "``dis``"
   CELLSIZE, Specify a cell size to be used, :func:`imod.util.empty_2d`, "``dx``, ``dy``"
   I3D, Specify whether the GEN file needs to be transformed to 3D (a vertical polygon), :class:`3D HFB <imod.mf6.HorizontalFlowBarrierResistance>`,
   IDF_TOP, The uppermost values of the snapped vertical polygon, :func:`linestring_to_square_zpolygons <imod.prepare.linestring_to_square_zpolygons>`, ``barrier_ztop``
   IDF_BOT, The lowermost values of the snapped vertical polygon, :func:`linestring_to_square_zpolygons <imod.prepare.linestring_to_square_zpolygons>`, ``barrier_zbot``

IMODPATH
********

The function IMODPATH computes flowlines based on the budget terms that result
from the iMODFLOW computation. The equivalent functionality in MODFLOW6 is the
particle tracking (PRT) model. This is currently not supported in iMOD Python.

ISGGRID
*******

Use this function to rasterize the selected ISG-files into IDF-files that can be
used by iMODFLOW in a runfile. There currently is no equivalent functionality in
iMOD Python to read and grid ISG files.

MF6TOIDF
********

Use this post-processing function to convert standard MODFLOW6 output to IDF
files. The eequivalent functionality in iMOD Python is mostly covered by the
following functions: :func:`imod.mf6.open_hds`, :func:`imod.mf6.open_cbc`,
:meth:`imod.mf6.Modflow6Simulation.open_flow_budget`,
:meth:`imod.mf6.Modflow6Simulation.open_transport_budget`,
:meth:`imod.mf6.Modflow6Simulation.open_head`,
:meth:`imod.mf6.Modflow6Simulation.open_concentration`. The table below lists pointers to
the functions and arguments that can be used to achieve full feature parity with
the iMOD5 BATCH function. You can write the data to IDF files using
:func:`imod.formats.idf.save`.

.. csv-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 5, 20, 20, 20

   BATCH argument, description, iMOD Python, argument
   ISTEADY, Indicates that the first entry in the output file is a steady-state solution, , 
   SDATE, The initial date of the model, ":func:`imod.mf6.open_hds`, :func:`imod.mf6.open_cbc`", ``simulation_start_time``
   DATEFORMAT, Enforce a long date format in the produced IDF filename, :func:`imod.formats.idf.save`, ``pattern``
   IDF, IDF file used as spatial definition,":func:`imod.mf6.open_hds`, :func:`imod.mf6.open_cbc`", ``grb_path``
   GRB, GRB file used as spatial definition,":func:`imod.mf6.open_hds`, :func:`imod.mf6.open_cbc`", ``grb_path``
   HED, The output file (\*.HED) for MODFLOW6 that contains the heads, :func:`imod.mf6.open_hds`, ``hds_path``
   BDG, The output file (\*.CBC) for MODFLOW6 that contains the flow budget, :func:`imod.mf6.open_cbc`, ``cbc_path``
   BDGUZF, The output file (\*.CBC) for MODFLOW6 that contains the UZF flow budgets, :func:`imod.mf6.open_cbc`, ``cbc_path``
   WC_UZF, The output file (\*.WC) for MODFLOW6 that contains the UZF water content, :func:`imod.mf6.open_dvs`, ``dvs_path``
   IDOUBLE, Save in double precision, `xarray.DataArray.astype <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.astype.html>`_, ``dtype=np.float32``
   SAVE\*, Save the results per layer, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, "``layer=[i]``"
   IPHRLVL, Save the first active value in the vertical dimension, ":func:`get_upper_active_grid_cells <imod.select.get_upper_active_grid_cells>`, :func:`get_upper_active_layer_number <imod.select.get_upper_active_layer_number>`", 
   IFILLHEAD, Fill head values where ``idomain==-1``, :func:`imod.prepare.fill`, ``dims='layer'``

Some settings previously configurable in iMOD5 are fixed in iMOD Python:

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python
   ISAVEENDDATE=1, Set time stamp to match the end of each time step, :func:`imod.mf6.open_hds`, 

GXG
***

Computes the GXG values, this is an indicator used in the Netherlands to
indicate the seasonal variation of the groundwater head. You can compute the GXG
values using the :func:`imod.evaluate.calculate_gxg` function. The table below
lists pointers to the functions and arguments that can be used to achieve full
feature parity with the iMOD5 BATCH function. See the API examples in
:func:`imod.evaluate.calculate_gxg` how to do similar things as with the GXG
function in iMOD5.

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python, argument
   ILAYER, Layers numbers to be used in the GxG computation, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``layer=[i]``
   NDIR, Number of folders to be processed, unnecessary in iMOD Python,
   SOURCEDIR, The folder and first part of the file name for all files that need to be used, :func:`imod.formats.idf.open`, ``pattern``
   OUTPUTFOLDER, The folder where the output files need to be written, :func:`imod.formats.idf.save`, ``directory``
   SURFACEIDF, "The IDF file that contains the surface elevation, if absent GXG is computed their reference", :func:`calculate_gxg <imod.evaluate.calculate_gxg>`, see API examples
   SYEAR, The start year (yyyy) for which IDF-files are used, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   EYEAR, The end year (yyyy) for which IDF-files are used, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   IYEAR, The year (yyyy) for which the GXG values need to be computed, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   STARTMONTH, The start month from the which the hydrological year starts, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   IPERIOD, Enter two integers for each month to express the inclusion of the first and second day of that particular month in the GXG computation, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   GENFILE, Enter a GEN-filename for polygon(s) for which mean values need to be computed, :func:`zonal_aggregate_raster <imod.prepare.zonal_aggregate_raster>`, 
   IDFNAME, "Cells in the IDF-file that are not equal to the NoDataValue of that IDF-file, the GXG will be computed.", `xarray.DataArray.where <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.where.html>`_, ``cond=idfdata.notnull()``

Some settings previously configurable in iMOD5 are fixed in iMOD Python:

.. csv-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 5, 20, 20

   BATCH argument, description, iMOD Python
   FIRSTDAY=14, First day in a month to include in the computation of the GXG, :func:`imod.evaluate.calculate_gxg`
   SECONDARY=28, Second day in a month to include in the computation of the GXG, :func:`imod.evaluate.calculate_gxg`
   ISEL=1, All active cells are used in the computation of the GXG, :func:`imod.evaluate.calculate_gxg`

MKWELLIPF
*********

The MKWELLIPF function computes the extraction strength for each well based on a
weighed value according to their length and permeability of the penetrated model
layer. Most of this functionality is implemented in iMOD Python's
:func:`assign_wells <imod.prepare.assign_wells>`. Note: The function computes rates for
each timestep in a timeseries, instead of averaging them. This function is also
called when running :meth:`imod.mf6.Well.to_mf6_pkg` and
:meth:`imod.mf6.LayeredWell.to_mf6_pkg`. The table below lists pointers to the
functions and arguments that can be used to achieve full feature parity with the
iMOD5 BATCH function.

.. csv-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 5, 20, 20, 20

   BATCH argument, description, iMOD Python, argument
   NIPF, Number of IPF files to be processed, unnecessary in iMOD Python, 
   IPF\{i\}, Name of the ith IPF file, :func:`imod.formats.ipf.read`, ``path``
   IXCOL, Column number of the x-coordinate, :func:`assign_wells <imod.prepare.assign_wells>`, ``dataframe["x"]``
   IYCOL, Column number of the y-coordinate, :func:`assign_wells <imod.prepare.assign_wells>`, ``dataframe["y"]``
   IQCOL, Column number of the well extraction rate, :func:`assign_wells <imod.prepare.assign_wells>`, ``dataframe["rate"]``
   ITCOL, Column number of the well filter top, :func:`assign_wells <imod.prepare.assign_wells>`, ``dataframe["top"]``
   IBCOL, Column number of the well filter bottom, :func:`assign_wells <imod.prepare.assign_wells>`, ``dataframe["bottom"]``
   ISS, Whether rates need to be averaged for a specific time instead of the complete time series, , 
   SDATE, The start date to be averaged, `pd.DataFrame.loc <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html>`_ , ``dataframe.loc[ dataframe["time"] >= start_time]``
   EDATE, The end date to be averaged, `pd.DataFrame.loc <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html>`_ , ``dataframe.loc[ dataframe["time"] <= start_time]``
   HNODATA, NoDataValue for the extraction rate, `pd.DataFrame.loc <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html>`_ , ``dataframe.loc[ dataframe["rate"] != nodata_value]``
   NLAY, Number of layers from which wells may be organized, unnecessary in iMOD Python, 
   TOPIDF\{i\}, Name of the ith IDF file that contains the top layer, :func:`assign_wells <imod.prepare.assign_wells>`, ``top``
   BOTIDF\{i\}, Name of the ith IDF file that contains the top layer, :func:`assign_wells <imod.prepare.assign_wells>`, ``bottom``
   KHKVIDF\{i\}, Name of the ith IDF file that contains the horizontal permeability, :func:`assign_wells <imod.prepare.assign_wells>`, ``k``
   KDVIDF\{i\}, Name of the ith IDF file that contains the transmissivity, not applicable for MODFLOW 6,
   MINKHT, Minimal horizontal permeability that will receive a well, :func:`assign_wells <imod.prepare.assign_wells>`, ``minimum_k``
   MINKD, Minimal transmissivity that will receive a well, not applicable for MODFLOW 6,
   FNODATA, NoDataValue for the top and bottom of the well screen (ITCOL and IBCOL),`pd.DataFrame.loc <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html>`_, ``dataframe.loc[ dataframe["top"] != fnodata]``

Some settings previously configurable in iMOD5 are fixed in iMOD Python:

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python
   IMIDF=0, How to compute midpoints of well screens when either top or bottom is missing, :func:`assign_wells <imod.prepare.assign_wells>`

GENPUZZLE
*********

The GENPUZZLE function reads a GEN file and creates a new GEN-file in which all
loose-ends are connected to form a continuous segment. This can be done with the
`line_merge function in geopandas.
<https://geopandas.org/en/latest/docs/reference/api/geopandas.GeoSeries.line_merge.html>`_

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python, argument
   GENFILE_IN, Give a GEN file containing x and y coordinates of GEN segments, :func:`imod.formats.gen.read`, ``path``
   GENFILE_OUT, Specify the output GEN file, :func:`imod.formats.gen.write`, ``path``

Some settings previously configurable in iMOD5 are fixed in iMOD Python:

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python
   IBINARY, Enforce GENFILE_OUT to be binary at all times, :func:`imod.formats.gen.write`

IDFSCALE
********

Rescale IDF-files according to different methodologies. This functionality is
implemented in iMOD Python for MODFLOW 6 simulations
:meth:`imod.mf6.Modflow6Simulation.regrid_like`. This automatically selects
default values for different packages. An overview of these is presented in
:doc:`../user-guide/08-regridding`. For individual grids, you can `call the
regridding functionality xugrid
<https://deltares.github.io/xugrid/examples/regridder_overview.html>`_, which
also works for structured grids, like saved in IDF files. Multiple methods are
available to upscale and downscale models. As upscaling methods SCLTYPE_UP 1 to
10 method are supported. The table below lists pointers to the functions and
arguments that can be used to achieve full feature parity with the iMOD5 BATCH
function.

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python, argument
   SCALESIZE, Cell size of the upscaled or downscaled IDF-file, :func:`imod.util.empty_2d`, "``dx``, ``dy``"
   SCLTYPE_UP, Method to upscale data, e.g. :meth:`NPF.regrid_like <imod.mf6.NodePropertyFlow.regrid_like>`, ``regridder_types``
   SCLTYPE_DOWN, Method to downscale data, e.g. :meth:`NPF.regrid_like <imod.mf6.NodePropertyFlow.regrid_like>`, ``regridder_types``
   SOURCEIDF, IDF file that needs to be rescaled, :func:`imod.formats.idf.open`, ``path``
   OUTFILE, IDF file that will be created, :func:`imod.formats.idf.save`, ``path``
   PERCENTILE, Percentile to be used for upscaling, `xugrid create_percentile_method <https://deltares.github.io/xugrid/api/xugrid.OverlapRegridder.html#xugrid.OverlapRegridder>`_, ``percentile``
   WEIGHFACTOR, Weight factor, :class:`imod.util.RegridderWeightsCache` , 
   WINDOW, Window to be used for rescaling, :meth:`mf6_sim.clip_box <imod.mf6.Modflow6Simulation.clip_box>`, "``x_min``, ``x_max``, ``y_min``, ``y_max``"


Some settings previously configurable in iMOD5 are fixed in iMOD Python:

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python
   BLOCK=4, size of the interpolation block, `xugrid.BaryCentricInterpolator <https://deltares.github.io/xugrid/api/xugrid.BarycentricInterpolator.html>`_


IDFTIMESERIE
************

Generate timeseries out of IDF-files that have the notation
``{item}_yyyymmdd_l{ilay}.idf``. These are IDF-files that yield from a normal
iMODFLOW simulation. Equivalent functionality in iMOD Python is found in
:func:`imod.select.points_values`.

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python, argument
   IPF1, Name of the IPF file that contains the locations, :func:`imod.select.points_values`, ``**points``
   IPF2, Name of the IPF file to store the timeseries, :func:`imod.formats.ipf.save`, ``path``
   ILAY, Layer number to be used in the timeseries, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``layer=[i]``
   SOURCEDIR, directory name of the folder that contains the specific files + the first (similar) part of the name of the files, :func:`imod.formats.idf.open`, ``path``
   SDATE, Start date of the timeseries, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   EDATE, Start date of the timeseries, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   LABELCOL, Column number of the label for the associated text files, :func:`imod.formats.ipf.save`, ``assoc_columns``
   NGXG, compute GXG values, :func:`imod.evaluate.calculate_gxg_points`, 
   ICLEAN, remove points from the IPF1 that are outside the domain of IDF data in SOURCEDIR, :func:`imod.select.points_values`, ``out_of_bounds``

Some settings previously configurable in iMOD5 are fixed in iMOD Python:

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python
   IASSF=0, specify whether or not to include the time series from IPF1 into the new IPF2 associated files, :func:`imod.select.points_values`
   TXTCOL=2, Enter the column to be used from the associated text files, :func:`imod.formats.ipf.read`
   INT=0, Carry out a 4-point polynomial interpolation of the 4 enclosed grid points surrounding the location of the measurement. Default INT=0 and it takes grid value, :func:`imod.select.points_values`


CREATSOF
********

The iMOD-Batch function SOF (Surface Overland Flow) is able to compute “spill”
levels (surface overlandflow levels) for large regions with or without supplied
outflow or outlet locations. iMOD Python does not have a direct equivalent
functionality, but you can use the `python package PyFlwDir
<https://deltares.github.io/pyflwdir/latest/quickstart.html>`_ to do similar
things.

IDFMEAN
*******

Compute a new IDF-file with the mean value (or minimum, maximum value) of
different IDF-files. This can be easily done with xarray:

.. code:: python

  import imod
  import xarray as xr

  idf_data = imod.formats.idf.open("path/to/your/idf/files/*.idf")
  mean_value_idf = idf_data.mean()  # or min, max, sum
  # To get data in the same shape as the original IDF files, you can use:
  mean_idf = xr.ones_like(idf_data) * mean_value_idf
  # Save the mean IDF file
  imod.formats.idf.save(mean_idf, "path/to/your/mean_idf_file.idf")

IDFMERGE
********

The MERGE function can be used to merge different IDF-files into a new IDF-file.
If these IDF-files might overlap, an interpolation between the overlapping
IDF-files will be carried out (if selected). Equivalent functionality in
iMOD Python is found in :func:`imod.formats.idf.open_subdomains`

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python, argument
   NMERGE, Number of IDF files to be merged, unnecessary in iMOD Python,
   SOURCEDIR, The folder and first part of the file name for all files that need to be merged, :func:`open_subdomains <imod.formats.idf.open_subdomains>`, ``path`` \& ``pattern``
   TARGETIDF, The IDF file that will be created, :func:`imod.formats.idf.save`, ``path``
   WINDOW, The window that needs to be computed solely, :meth:`mf6_sim.clip_box <imod.mf6.Modflow6Simulation.clip_box>`, "``x_min``, ``x_max``, ``y_min``, ``y_max``"
   MASKIDF, IDF-file to mask areas , `xarray.DataArray.where <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.where.html>`_, ``cond=maskidf.notnull()``

Some settings previously configurable in iMOD5 are fixed in iMOD Python:

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python
   IINT=0, Specify IINT=1 to enforce a smooth interpolation between IDF-files in their overlapping areas, this is the default setting., :func:`open_subdomains <imod.formats.idf.open_subdomains>`

PLOT
****

The PLOT function can be used to construct figures that are normally displayed
on the graphical display of iMOD. Equivalent functionality in iMOD Python is
found in :func:`imod.visualize.plot_map`.

WBALANCE
********

The WBALANCE function calculates the water balance based on the model output for
the steady state condition or for a specific period and area. The result is a
CSV file (Step 1). Alternatively, this function can create images, IDF files
and/or CVS files from aggregation on existing CSV files (Step 2). The hardest
part of the functionality for step 1 in iMOD Python is found in
:func:`imod.evaluate.facebudget`, for ISEL=3-type of behaviour of the
facebudgets. Other water balance terms can be computed by using general xarray
logic, for examples see below. To plot waterbalance terms, you can use
:func:`imod.visualize.waterbalance_barchart`.


.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python, argument
   NDIR, Number of folders to be processed, unnecessary in iMOD Python,
   SOURCEDIR, The folder and first part of the file name for all files that need to be used, :func:`imod.formats.idf.open`, ``path`` \& ``pattern``
   ILAYER, Layer numbers to be used in the water balance computation, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``layer=[i]``
   SDATE, The start date of the water balance computation, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   EDATE, The end date of the water balance computation, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   IYEAR, The year for which the water balance needs to be computed, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   NPERIOD, The number of periods for which the water balance needs to be computed, unnecessary in iMOD Python,
   PERIOD{i}, The period for which the water balance needs to be computed, `xarray.DataArray.sel <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html>`_, ``time=...``
   NBAL, The number of budget terms to be computed, unnecessary in iMOD Python,
   BAL{i}, The budget term to be computed, :func:`imod.evaluate.facebudget`,"``front``, ``lower``, ``right``"
   BAL{i}SYS, The number of systems to be included in the water balance,"unnecessary in iMOD Python, as it treats systems as separate packages",
   ISEL,"The type of water balance to be computed, see below for the different types of ISEL", See examples below,
   GENFILE, The GEN file that contains the area for which the water balance needs to be computed, :func:`imod.formats.gen.read`, ``path``
   IDFNAME, The IDF file that contains the area for which the water balance needs to be computed, :func:`imod.formats.idf.open`, ``path``

Some settings previously configurable in iMOD5 are fixed in iMOD Python:

.. csv-table::
   :header-rows: 1
   :stub-columns: 1

   BATCH argument, description, iMOD Python
   WBEX=0, Compute interconnected fluxes between zones, :func:`imod.evaluate.facebudget`


To get ISEL=1-type of behaviour, you can use:

.. code:: python

  import imod

  cbc_data = imod.mf6.open_cbc("path/to/your/cbc_file.cbc", merge_to_dataset=True)
  wbalance = cbc_data.sum()
  # to compute  per timestep
  wbalance_per_time = cbc_data.sum(dim=("layer", "y", "x"))

Or to get ISEL=2-type of behaviour, you can use:

.. code:: python

  import imod
  import xarray as xr

  cbc_data = imod.mf6.open_cbc("path/to/your/cbc_file.cbc", "path/to/your/grb_file.grb")
  gen_data = imod.formats.gen.read("path/to/your/gen_file.gen")
  
  # Pop the facebudgets from the cbc_data
  front = cbc_data.pop("flow-front-face")
  lower = cbc_data.pop("flow-lower-face")
  right = cbc_data.pop("flow-right-face")

  like = front.isel(layer=0, time=0, drop=True)
  wbalance_area = imod.prepare.rasterize(gen_data, like)
  netflow = imod.evaluate.facebudget(
      wbalance_area,
      front=front,
      lower=lower,
      right=right,
      netflow=True
  )
  cbc_data["netflow"] = netflow
  # Convert cbc_data from dict to xarray.Dataset
  cbc_data = xr.merge([cbc_data])
  # Mask all budget terms that are not present in the cbc_data
  cbc_area = cbc_data.where(wbalance_area.notnull())
  # Sum the budget terms for waterbalance of the area
  cbc_area.sum()

Or to get ISEL=3-type of behaviour, you can use:

.. code:: python

  import imod
  import xarray as xr

  cbc_data = imod.mf6.open_cbc("path/to/your/cbc_file.cbc", "path/to/your/grb_file.grb")
  idf_data = imod.formats.idf.open("path/to/your/idf_file.idf")
  
  # Pop the facebudgets from the cbc_data
  front = cbc_data.pop("flow-front-face")
  lower = cbc_data.pop("flow-lower-face")
  right = cbc_data.pop("flow-right-face")

  like = front.isel(layer=0, time=0, drop=True)
  netflow = imod.evaluate.facebudget(
      idf_data,
      front=front,
      lower=lower,
      right=right,
      netflow=True
  )
  cbc_data["netflow"] = netflow
  # Convert cbc_data from dict to xarray.Dataset
  cbc_data = xr.merge([cbc_data])
  # Mask all budget terms that are not present in the cbc_data
  cbc_area = cbc_data.where(idf_data.notnull())
  # Sum the budget terms for waterbalance of the area
  cbc_area.sum()
  