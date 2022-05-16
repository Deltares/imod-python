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
    GroundwaterFlowModel

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

Boundary Conditions
-------------------

.. autosummary::
    :toctree: generated/mf6

    ConstantHead
    Drainage
    Evapotranspiration
    GeneralHeadBoundary
    InitialConditions
    NodePropertyFlow
    Recharge
    River
    SpecificStorage
    StorageCoefficient
    UnsaturatedZoneFlow
    WellDisStructured
    WellDisVertices
