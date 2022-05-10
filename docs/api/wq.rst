.. currentmodule:: imod.wq

iMOD-WQ
=======

Model
-----

.. autosummary::
    :toctree: generated/wq
    
    SeawatModel
    SeawatModel.time_discretization

Settings
--------

.. autosummary::
    :toctree: generated/wq

    TimeDiscretization
    OutputControl
    PreconditionedConjugateGradientSolver
    GeneralizedConjugateGradientSolver
    ParallelKrylovFlowSolver
    ParallelKrylovTransportSolver

Flow
----

.. autosummary::
    :toctree: generated/wq

    BasicFlow
    ConstantHead
    Drainage
    EvapotranspirationTopLayer
    EvapotranspirationLayers
    EvapotranspirationHighestActive
    GeneralHeadBoundary
    LayerPropertyFlow
    RechargeTopLayer
    RechargeLayers
    RechargeHighestActive
    River
    Well
    VariableDensityFlow

Transport
---------

.. autosummary::
    :toctree: generated/wq

    AdvectionTVD
    AdvectionMOC
    AdvectionModifiedMOC
    AdvectionHybridMOC
    AdvectionFiniteDifference
    Dispersion
    MassLoading
    TimeVaryingConstantConcentration
