.. currentmodule:: imod.mf6

MODFLOW 6
=========

Read Output
-----------

.. autosummary::
    :toctree: generated/mf6

    open_hds
    open_cbc
    read_cbc_headers
    open_dvs

Model objects & methods
-----------------------

.. autosummary::
    :toctree: generated/mf6

    Modflow6Simulation
    Modflow6Simulation.create_time_discretization
    Modflow6Simulation.copy
    Modflow6Simulation.write
    Modflow6Simulation.dump
    Modflow6Simulation.run
    Modflow6Simulation.mask_all_models
    Modflow6Simulation.open_flow_budget
    Modflow6Simulation.open_transport_budget
    Modflow6Simulation.open_head
    Modflow6Simulation.open_concentration
    Modflow6Simulation.from_file
    Modflow6Simulation.from_imod5_data
    Modflow6Simulation.clip_box
    Modflow6Simulation.split
    Modflow6Simulation.regrid_like
    Modflow6Simulation.is_split
    Modflow6Simulation.get_exchange_relationships
    Modflow6Simulation.get_models_of_type
    Modflow6Simulation.get_models
    GroundwaterFlowModel
    GroundwaterFlowModel.mask_all_packages
    GroundwaterFlowModel.prepare_wel_for_mf6
    GroundwaterFlowModel.dump
    GroundwaterFlowModel.from_imod5_data
    GroundwaterFlowModel.clip_box
    GroundwaterFlowModel.from_file
    GroundwaterFlowModel.purge_empty_packages
    GroundwaterFlowModel.is_splitting_supported
    GroundwaterFlowModel.is_regridding_supported
    GroundwaterFlowModel.is_clipping_supported
    GroundwaterFlowModel.validate
    GroundwaterFlowModel.set_newton
    GroundwaterFlowModel.write
    GroundwaterTransportModel
    GroundwaterTransportModel.mask_all_packages
    GroundwaterTransportModel.dump
    GroundwaterTransportModel.clip_box
    GroundwaterTransportModel.from_file
    GroundwaterTransportModel.purge_empty_packages
    GroundwaterTransportModel.is_splitting_supported
    GroundwaterTransportModel.is_regridding_supported
    GroundwaterTransportModel.is_clipping_supported
    GroundwaterTransportModel.validate
    GroundwaterTransportModel.set_newton
    GroundwaterTransportModel.write

Discretization
--------------

.. autosummary::
    :toctree: generated/mf6

    StructuredDiscretization
    StructuredDiscretization.regrid_like
    StructuredDiscretization.from_imod5_data
    StructuredDiscretization.clip_box
    VerticesDiscretization
    VerticesDiscretization.regrid_like
    VerticesDiscretization.clip_box
    TimeDiscretization

Model settings
--------------

.. autosummary::
    :toctree: generated/mf6

    OutputControl
    Solution
    SolutionPresetSimple
    SolutionPresetModerate
    SolutionPresetComplex
    ValidationSettings

Flow Packages
-------------

.. autosummary::
    :toctree: generated/mf6

    ApiPackage
    Buoyancy
    ConstantHead
    ConstantHead.from_imod5_data
    ConstantHead.from_imod5_shd_data
    ConstantHead.mask
    ConstantHead.regrid_like
    ConstantHead.clip_box
    Drainage
    Drainage.mask
    Drainage.from_imod5_data
    Drainage.regrid_like
    Drainage.cleanup
    Drainage.clip_box
    Drainage.reallocate
    Evapotranspiration
    Evapotranspiration.mask
    Evapotranspiration.regrid_like
    Evapotranspiration.clip_box
    GeneralHeadBoundary
    GeneralHeadBoundary.from_imod5_data
    GeneralHeadBoundary.mask
    GeneralHeadBoundary.regrid_like
    GeneralHeadBoundary.cleanup
    GeneralHeadBoundary.clip_box
    GeneralHeadBoundary.reallocate
    HorizontalFlowBarrierHydraulicCharacteristic
    HorizontalFlowBarrierHydraulicCharacteristic.clip_box
    HorizontalFlowBarrierHydraulicCharacteristic.to_mf6_pkg
    HorizontalFlowBarrierHydraulicCharacteristic.snap_to_grid
    HorizontalFlowBarrierMultiplier
    HorizontalFlowBarrierMultiplier.clip_box
    HorizontalFlowBarrierMultiplier.to_mf6_pkg
    HorizontalFlowBarrierMultiplier.snap_to_grid
    HorizontalFlowBarrierResistance
    HorizontalFlowBarrierResistance.to_mf6_pkg
    HorizontalFlowBarrierResistance.clip_box
    HorizontalFlowBarrierResistance.snap_to_grid
    LayeredWell
    LayeredWell.from_imod5_data
    LayeredWell.from_imod5_cap_data
    LayeredWell.mask
    LayeredWell.regrid_like
    LayeredWell.to_mf6_pkg
    LayeredWell.clip_box
    InitialConditions
    InitialConditions.from_imod5_data
    InitialConditions.mask
    InitialConditions.regrid_like
    InitialConditions.clip_box
    NodePropertyFlow
    NodePropertyFlow.from_imod5_data
    NodePropertyFlow.mask
    NodePropertyFlow.regrid_like
    NodePropertyFlow.clip_box
    Recharge
    Recharge.from_imod5_data
    Recharge.from_imod5_cap_data
    Recharge.mask
    Recharge.regrid_like
    Recharge.clip_box
    Recharge.reallocate
    River
    River.from_imod5_data
    River.mask
    River.regrid_like
    River.cleanup
    River.clip_box
    River.reallocate
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.to_mf6_pkg
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.clip_box
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.snap_to_grid
    SingleLayerHorizontalFlowBarrierMultiplier
    SingleLayerHorizontalFlowBarrierMultiplier.to_mf6_pkg
    SingleLayerHorizontalFlowBarrierMultiplier.clip_box
    SingleLayerHorizontalFlowBarrierMultiplier.snap_to_grid
    SingleLayerHorizontalFlowBarrierResistance
    SingleLayerHorizontalFlowBarrierResistance.from_imod5_data
    SingleLayerHorizontalFlowBarrierResistance.to_mf6_pkg
    SingleLayerHorizontalFlowBarrierResistance.clip_box
    SingleLayerHorizontalFlowBarrierResistance.snap_to_grid
    SpecificStorage
    SpecificStorage.mask
    SpecificStorage.regrid_like
    SpecificStorage.clip_box
    StorageCoefficient
    StorageCoefficient.from_imod5_data
    StorageCoefficient.mask
    StorageCoefficient.regrid_like
    StorageCoefficient.clip_box
    UnsaturatedZoneFlow
    Well
    Well.cleanup
    Well.from_imod5_data
    Well.mask
    Well.regrid_like
    Well.to_mf6_pkg
    Well.clip_box


Transport Packages
------------------

.. autosummary::
    :toctree: generated/mf6

    ApiPackage
    AdvectionCentral
    AdvectionTVD
    AdvectionUpstream
    ConstantConcentration
    Dispersion
    ImmobileStorageTransfer
    MobileStorageTransfer
    MassSourceLoading
    SourceSinkMixing
    SourceSinkMixing.from_flow_model


.. currentmodule:: imod.mf6.regrid

Regrid
======

Regrid Method Settings
----------------------

.. autosummary::
    :toctree: generated/mf6/regrid

    ConstantHeadRegridMethod
    DiscretizationRegridMethod
    DispersionRegridMethod
    DrainageRegridMethod
    EmptyRegridMethod
    EvapotranspirationRegridMethod
    GeneralHeadBoundaryRegridMethod
    InitialConditionsRegridMethod
    MobileStorageTransferRegridMethod
    NodePropertyFlowRegridMethod
    RechargeRegridMethod
    RiverRegridMethod
    SpecificStorageRegridMethod
    StorageCoefficientRegridMethod