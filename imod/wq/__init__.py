"""
Create Water Quality model.

Create an :class:`imod.wq.SeawatModel` and add desired packages to the model
(e.g. :class:`imod.wq.Well`, :class:`imod.wq.Dispersion`). See :doc:`/examples`
and :doc:`/model` for workflow documentation.
"""

# make classes directly available under imod.wq
from imod.wq.adv import (
    AdvectionFiniteDifference,
    AdvectionHybridMOC,
    AdvectionMOC,
    AdvectionModifiedMOC,
    AdvectionTVD,
)
from imod.wq.bas import BasicFlow
from imod.wq.btn import BasicTransport
from imod.wq.chd import ConstantHead
from imod.wq.dis import TimeDiscretization
from imod.wq.drn import Drainage
from imod.wq.dsp import Dispersion
from imod.wq.evt import (
    EvapotranspirationHighestActive,
    EvapotranspirationLayers,
    EvapotranspirationTopLayer,
)
from imod.wq.ghb import GeneralHeadBoundary
from imod.wq.lpf import LayerPropertyFlow
from imod.wq.mal import MassLoading
from imod.wq.model import SeawatModel
from imod.wq.oc import OutputControl
from imod.wq.rch import RechargeHighestActive, RechargeLayers, RechargeTopLayer
from imod.wq.riv import River
from imod.wq.slv import (
    GeneralizedConjugateGradientSolver,
    ParallelKrylovFlowSolver,
    ParallelKrylovTransportSolver,
    PreconditionedConjugateGradientSolver,
)
from imod.wq.tvc import TimeVaryingConstantConcentration
from imod.wq.vdf import VariableDensityFlow
from imod.wq.wel import Well
