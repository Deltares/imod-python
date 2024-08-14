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
    Drainage
    Evapotranspiration
    GeneralHeadBoundary
    HorizontalFlowBarrierHydraulicCharacteristic
    HorizontalFlowBarrierMultiplier
    HorizontalFlowBarrierResistance
    InitialConditions
    NodePropertyFlow
    Recharge
    River
    SpecificStorage
    StorageCoefficient
    UnsaturatedZoneFlow
    Well
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