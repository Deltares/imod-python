.. currentmodule:: imod.mf6

MODFLOW6
========

Read Output
-----------

.. autosummary::
    :toctree: generated/mf6

    open_hds
    open_cbc
    read_cbc_headers

Model objects & methods
-----------------------

.. autosummary::
    :toctree: generated/mf6

    Modflow6Simulation
    Modflow6Simulation.create_time_discretization
    Modflow6Simulation.write
    Modflow6Simulation.dump
    Modflow6Simulation.run
    Modflow6Simulation.open_flow_budget
    Modflow6Simulation.open_transport_budget
    Modflow6Simulation.open_head
    Modflow6Simulation.open_concentration
    Modflow6Simulation.clip_box
    Modflow6Simulation.split
    Modflow6Simulation.regrid_like
    GroundwaterFlowModel
    GroundwaterFlowModel.mask_all_packages
    GroundwaterFlowModel.dump
    GroundwaterTransportModel
    GroundwaterTransportModel.mask_all_packages
    GroundwaterTransportModel.dump

Discretization
--------------

.. autosummary::
    :toctree: generated/mf6

    StructuredDiscretization
    VerticesDiscretization
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
    Drainage
    Drainage.mask
    Drainage.from_imod5_data
    Drainage.regrid_like
    Drainage.cleanup
    Evapotranspiration
    Evapotranspiration.mask
    Evapotranspiration.regrid_like
    GeneralHeadBoundary
    GeneralHeadBoundary.from_imod5_data
    GeneralHeadBoundary.mask
    GeneralHeadBoundary.regrid_like
    GeneralHeadBoundary.cleanup
    HorizontalFlowBarrierHydraulicCharacteristic
    HorizontalFlowBarrierHydraulicCharacteristic.to_mf6_pkg
    HorizontalFlowBarrierMultiplier
    HorizontalFlowBarrierMultiplier.to_mf6_pkg
    HorizontalFlowBarrierResistance
    HorizontalFlowBarrierResistance.to_mf6_pkg
    LayeredWell
    LayeredWell.from_imod5_data
    LayeredWell.mask
    LayeredWell.regrid_like
    LayeredWell.to_mf6_pkg
    InitialConditions
    InitialConditions.from_imod5_data
    InitialConditions.mask
    InitialConditions.regrid_like
    NodePropertyFlow
    NodePropertyFlow.from_imod5_data
    NodePropertyFlow.mask
    NodePropertyFlow.regrid_like
    Recharge
    Recharge.from_imod5_data
    Recharge.mask
    Recharge.regrid_like
    River
    River.from_imod5_data
    River.mask
    River.regrid_like
    River.cleanup
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic
    SingleLayerHorizontalFlowBarrierHydraulicCharacteristic.to_mf6_pkg
    SingleLayerHorizontalFlowBarrierMultiplier
    SingleLayerHorizontalFlowBarrierMultiplier.to_mf6_pkg
    SingleLayerHorizontalFlowBarrierResistance
    SingleLayerHorizontalFlowBarrierResistance.from_imod5_data
    SingleLayerHorizontalFlowBarrierResistance.to_mf6_pkg
    SpecificStorage
    SpecificStorage.mask
    SpecificStorage.regrid_like
    StorageCoefficient
    StorageCoefficient.from_imod5_data
    StorageCoefficient.mask
    StorageCoefficient.regrid_like
    UnsaturatedZoneFlow
    Well
    Well.cleanup
    Well.from_imod5_data
    Well.mask
    Well.regrid_like
    Well.to_mf6_pkg
    WellDisStructured
    WellDisVertices

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