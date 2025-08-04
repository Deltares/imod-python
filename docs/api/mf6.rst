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
    GroundwaterFlowModel.regrid_like
    GroundwaterFlowModel.dump
    GroundwaterFlowModel.from_imod5_data
    GroundwaterFlowModel.clip_box
    GroundwaterFlowModel.from_file
    GroundwaterFlowModel.purge_empty_packages
    GroundwaterFlowModel.is_use_newton
    GroundwaterFlowModel.validate
    GroundwaterFlowModel.set_newton
    GroundwaterFlowModel.write
    GroundwaterTransportModel
    GroundwaterTransportModel.mask_all_packages
    GroundwaterTransportModel.dump
    GroundwaterTransportModel.clip_box
    GroundwaterTransportModel.regrid_like
    GroundwaterTransportModel.from_file
    GroundwaterTransportModel.purge_empty_packages
    GroundwaterTransportModel.validate
    GroundwaterTransportModel.is_use_newton
    GroundwaterTransportModel.write

Discretization
--------------

.. autosummary::
    :toctree: generated/mf6

    StructuredDiscretization
    StructuredDiscretization.from_imod5_data
    StructuredDiscretization.get_regrid_methods
    StructuredDiscretization.regrid_like
    StructuredDiscretization.from_file
    StructuredDiscretization.to_netcdf
    StructuredDiscretization.copy
    StructuredDiscretization.clip_box
    StructuredDiscretization.mask
    StructuredDiscretization.is_empty
    StructuredDiscretization.write
    VerticesDiscretization
    VerticesDiscretization.get_regrid_methods
    VerticesDiscretization.regrid_like
    VerticesDiscretization.from_file
    VerticesDiscretization.to_netcdf
    VerticesDiscretization.copy
    VerticesDiscretization.clip_box
    VerticesDiscretization.mask
    VerticesDiscretization.is_empty
    VerticesDiscretization.write
    TimeDiscretization
    TimeDiscretization.write
    TimeDiscretization.from_file
    TimeDiscretization.to_netcdf
    TimeDiscretization.copy
    TimeDiscretization.clip_box
    TimeDiscretization.mask
    TimeDiscretization.regrid_like
    TimeDiscretization.is_empty
    TimeDiscretization.get_regrid_methods

Model settings
--------------

.. autosummary::
    :toctree: generated/mf6

    OutputControl
    OutputControl.is_budget_output
    OutputControl.write
    OutputControl.from_file
    OutputControl.to_netcdf
    OutputControl.copy
    OutputControl.clip_box
    OutputControl.mask
    OutputControl.regrid_like
    OutputControl.is_empty
    OutputControl.get_regrid_methods
    Solution
    Solution.write
    Solution.from_file
    Solution.to_netcdf
    Solution.copy
    Solution.clip_box
    Solution.mask
    Solution.regrid_like
    Solution.is_empty
    Solution.get_regrid_methods
    SolutionPresetSimple
    SolutionPresetModerate
    SolutionPresetComplex
    ValidationSettings

Flow Packages
-------------

.. autosummary::
    :toctree: generated/mf6

    ApiPackage
    ApiPackage.write
    ApiPackage.from_file
    ApiPackage.to_netcdf
    ApiPackage.copy
    ApiPackage.is_empty
    ApiPackage.get_regrid_methods
    Buoyancy
    Buoyancy.write
    Buoyancy.from_file
    Buoyancy.to_netcdf
    Buoyancy.copy
    Buoyancy.is_empty
    Buoyancy.get_regrid_methods
    ConstantHead
    ConstantHead.from_imod5_data
    ConstantHead.from_imod5_shd_data
    ConstantHead.mask
    ConstantHead.regrid_like
    ConstantHead.clip_box
    ConstantHead.write
    ConstantHead.from_file
    ConstantHead.to_netcdf
    ConstantHead.copy
    ConstantHead.is_empty
    ConstantHead.get_regrid_methods
    Drainage
    Drainage.mask
    Drainage.from_imod5_data
    Drainage.regrid_like
    Drainage.cleanup
    Drainage.clip_box
    Drainage.write
    Drainage.from_file
    Drainage.to_netcdf
    Drainage.copy
    Drainage.is_empty
    Drainage.get_regrid_methods
    Drainage.aggregate_layers
    Drainage.reallocate
    Evapotranspiration
    Evapotranspiration.mask
    Evapotranspiration.regrid_like
    Evapotranspiration.clip_box
    Evapotranspiration.write
    Evapotranspiration.from_file
    Evapotranspiration.to_netcdf
    Evapotranspiration.copy
    Evapotranspiration.is_empty
    Evapotranspiration.get_regrid_methods
    GeneralHeadBoundary
    GeneralHeadBoundary.from_imod5_data
    GeneralHeadBoundary.mask
    GeneralHeadBoundary.regrid_like
    GeneralHeadBoundary.cleanup
    GeneralHeadBoundary.clip_box
    GeneralHeadBoundary.write
    GeneralHeadBoundary.from_file
    GeneralHeadBoundary.to_netcdf
    GeneralHeadBoundary.copy
    GeneralHeadBoundary.is_empty
    GeneralHeadBoundary.get_regrid_methods
    GeneralHeadBoundary.aggregate_layers
    GeneralHeadBoundary.reallocate
    HorizontalFlowBarrierHydraulicCharacteristic
    HorizontalFlowBarrierHydraulicCharacteristic.clip_box
    HorizontalFlowBarrierHydraulicCharacteristic.to_mf6_pkg
    HorizontalFlowBarrierHydraulicCharacteristic.snap_to_grid
    HorizontalFlowBarrierHydraulicCharacteristic.write
    HorizontalFlowBarrierHydraulicCharacteristic.from_file
    HorizontalFlowBarrierHydraulicCharacteristic.to_netcdf
    HorizontalFlowBarrierHydraulicCharacteristic.copy
    HorizontalFlowBarrierHydraulicCharacteristic.is_empty
    HorizontalFlowBarrierHydraulicCharacteristic.get_regrid_methods
    HorizontalFlowBarrierMultiplier
    HorizontalFlowBarrierMultiplier.clip_box
    HorizontalFlowBarrierMultiplier.to_mf6_pkg
    HorizontalFlowBarrierMultiplier.snap_to_grid
    HorizontalFlowBarrierMultiplier.write
    HorizontalFlowBarrierMultiplier.from_file
    HorizontalFlowBarrierMultiplier.to_netcdf
    HorizontalFlowBarrierMultiplier.copy
    HorizontalFlowBarrierMultiplier.is_empty
    HorizontalFlowBarrierMultiplier.get_regrid_methods
    HorizontalFlowBarrierResistance
    HorizontalFlowBarrierResistance.to_mf6_pkg
    HorizontalFlowBarrierResistance.clip_box
    HorizontalFlowBarrierResistance.snap_to_grid
    HorizontalFlowBarrierResistance.write
    HorizontalFlowBarrierResistance.from_file
    HorizontalFlowBarrierResistance.to_netcdf
    HorizontalFlowBarrierResistance.copy
    HorizontalFlowBarrierResistance.is_empty
    HorizontalFlowBarrierResistance.get_regrid_methods
    LayeredWell
    LayeredWell.from_imod5_data
    LayeredWell.from_imod5_cap_data
    LayeredWell.mask
    LayeredWell.regrid_like
    LayeredWell.to_mf6_pkg
    LayeredWell.clip_box
    LayeredWell.write
    LayeredWell.from_file
    LayeredWell.to_netcdf
    LayeredWell.copy
    LayeredWell.is_empty
    LayeredWell.is_splitting_supported
    LayeredWell.is_regridding_supported
    LayeredWell.is_clipping_supported
    LayeredWell.is_grid_agnostic_package
    LayeredWell.get_regrid_methods
    InitialConditions
    InitialConditions.from_imod5_data
    InitialConditions.mask
    InitialConditions.regrid_like
    InitialConditions.clip_box
    InitialConditions.write
    InitialConditions.from_file
    InitialConditions.to_netcdf
    InitialConditions.copy
    InitialConditions.is_empty
    InitialConditions.get_regrid_methods
    NodePropertyFlow
    NodePropertyFlow.from_imod5_data
    NodePropertyFlow.mask
    NodePropertyFlow.regrid_like
    NodePropertyFlow.clip_box
    NodePropertyFlow.write
    NodePropertyFlow.from_file
    NodePropertyFlow.to_netcdf
    NodePropertyFlow.copy
    NodePropertyFlow.is_empty
    NodePropertyFlow.get_regrid_methods
    Recharge
    Recharge.from_imod5_data
    Recharge.from_imod5_cap_data
    Recharge.mask
    Recharge.regrid_like
    Recharge.clip_box
    Recharge.write
    Recharge.from_file
    Recharge.to_netcdf
    Recharge.copy
    Recharge.is_empty
    Recharge.get_regrid_methods
    Recharge.aggregate_layers
    Recharge.reallocate
    River
    River.from_imod5_data
    River.mask
    River.regrid_like
    River.cleanup
    River.clip_box
    River.write
    River.from_file
    River.to_netcdf
    River.copy
    River.is_empty
    River.get_regrid_methods
    River.aggregate_layers
    River.reallocate
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.to_mf6_pkg
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.clip_box
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.snap_to_grid
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.write
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.from_file
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.to_netcdf
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.copy
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.is_empty
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.get_regrid_methods
    SingleLayerHorizontalFlowBarrierMultiplier
    SingleLayerHorizontalFlowBarrierMultiplier.to_mf6_pkg
    SingleLayerHorizontalFlowBarrierMultiplier.clip_box
    SingleLayerHorizontalFlowBarrierMultiplier.snap_to_grid
    SingleLayerHorizontalFlowBarrierMultiplier.write
    SingleLayerHorizontalFlowBarrierMultiplier.from_file
    SingleLayerHorizontalFlowBarrierMultiplier.to_netcdf
    SingleLayerHorizontalFlowBarrierMultiplier.copy
    SingleLayerHorizontalFlowBarrierMultiplier.is_empty
    SingleLayerHorizontalFlowBarrierMultiplier.get_regrid_methods
    SingleLayerHorizontalFlowBarrierResistance
    SingleLayerHorizontalFlowBarrierResistance.from_imod5_data
    SingleLayerHorizontalFlowBarrierResistance.to_mf6_pkg
    SingleLayerHorizontalFlowBarrierResistance.clip_box
    SingleLayerHorizontalFlowBarrierResistance.snap_to_grid
    SingleLayerHorizontalFlowBarrierResistance.write
    SingleLayerHorizontalFlowBarrierResistance.from_file
    SingleLayerHorizontalFlowBarrierResistance.to_netcdf
    SingleLayerHorizontalFlowBarrierResistance.copy
    SingleLayerHorizontalFlowBarrierResistance.is_empty
    SingleLayerHorizontalFlowBarrierResistance.get_regrid_methods
    SpecificStorage
    SpecificStorage.mask
    SpecificStorage.regrid_like
    SpecificStorage.clip_box
    SpecificStorage.write
    SpecificStorage.from_file
    SpecificStorage.to_netcdf
    SpecificStorage.copy
    SpecificStorage.is_empty
    SpecificStorage.get_regrid_methods
    StorageCoefficient
    StorageCoefficient.from_imod5_data
    StorageCoefficient.mask
    StorageCoefficient.regrid_like
    StorageCoefficient.clip_box
    StorageCoefficient.write
    StorageCoefficient.from_file
    StorageCoefficient.to_netcdf
    StorageCoefficient.copy
    StorageCoefficient.is_empty
    StorageCoefficient.get_regrid_methods
    UnsaturatedZoneFlow
    UnsaturatedZoneFlow.clip_box
    UnsaturatedZoneFlow.mask
    UnsaturatedZoneFlow.regrid_like
    UnsaturatedZoneFlow.write
    UnsaturatedZoneFlow.from_file
    UnsaturatedZoneFlow.to_netcdf
    UnsaturatedZoneFlow.copy
    UnsaturatedZoneFlow.is_empty
    UnsaturatedZoneFlow.get_regrid_methods
    Well
    Well.cleanup
    Well.from_imod5_data
    Well.mask
    Well.regrid_like
    Well.to_mf6_pkg
    Well.clip_box
    Well.write
    Well.from_file
    Well.to_netcdf
    Well.copy
    Well.is_empty
    Well.get_regrid_methods


Transport Packages
------------------

.. autosummary::
    :toctree: generated/mf6

    ApiPackage
    ApiPackage.write
    ApiPackage.from_file
    ApiPackage.to_netcdf
    ApiPackage.copy
    ApiPackage.is_empty
    ApiPackage.get_regrid_methods
    AdvectionCentral
    AdvectionCentral.write
    AdvectionCentral.from_file
    AdvectionCentral.to_netcdf
    AdvectionCentral.copy
    AdvectionCentral.is_empty
    AdvectionCentral.get_regrid_methods
    AdvectionTVD
    AdvectionTVD.write
    AdvectionTVD.from_file
    AdvectionTVD.to_netcdf
    AdvectionTVD.copy
    AdvectionTVD.is_empty
    AdvectionTVD.get_regrid_methods
    AdvectionUpstream
    AdvectionUpstream.write
    AdvectionUpstream.from_file
    AdvectionUpstream.to_netcdf
    AdvectionUpstream.copy
    AdvectionUpstream.is_empty
    AdvectionUpstream.get_regrid_methods
    ConstantConcentration
    ConstantConcentration.write
    ConstantConcentration.from_file
    ConstantConcentration.to_netcdf
    ConstantConcentration.copy
    ConstantConcentration.is_empty
    ConstantConcentration.get_regrid_methods
    Dispersion
    Dispersion.write
    Dispersion.from_file
    Dispersion.to_netcdf
    Dispersion.copy
    Dispersion.is_empty
    Dispersion.get_regrid_methods
    ImmobileStorageTransfer
    ImmobileStorageTransfer.write
    ImmobileStorageTransfer.from_file
    ImmobileStorageTransfer.to_netcdf
    ImmobileStorageTransfer.copy
    ImmobileStorageTransfer.is_empty
    ImmobileStorageTransfer.get_regrid_methods
    MobileStorageTransfer
    MobileStorageTransfer.write
    MobileStorageTransfer.from_file
    MobileStorageTransfer.to_netcdf
    MobileStorageTransfer.copy
    MobileStorageTransfer.is_empty
    MobileStorageTransfer.get_regrid_methods
    MassSourceLoading
    MassSourceLoading.write
    MassSourceLoading.from_file
    MassSourceLoading.to_netcdf
    MassSourceLoading.copy
    MassSourceLoading.is_empty
    MassSourceLoading.get_regrid_methods
    SourceSinkMixing
    SourceSinkMixing.from_flow_model
    SourceSinkMixing.write
    SourceSinkMixing.from_file
    SourceSinkMixing.to_netcdf
    SourceSinkMixing.copy
    SourceSinkMixing.is_empty
    SourceSinkMixing.get_regrid_methods


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