# make classes directly available under imod.wq
from imod.wq.adv import AdvectionTVD, AdvectionModifiedMOC
from imod.wq.bas import BasicFlow
from imod.wq.btn import BasicTransport
from imod.wq.chd import ConstantHead
from imod.wq.dis import TimeDiscretization
from imod.wq.drn import Drainage
from imod.wq.dsp import Dispersion
from imod.wq.ghb import GeneralHeadBoundary
from imod.wq.lpf import LayerPropertyFlow
from imod.wq.oc import OutputControl
from imod.wq.model import SeawatModel
from imod.wq.rch import RechargeHighestActive, RechargeLayers, RechargeTopLayer
from imod.wq.riv import River
from imod.wq.slv import (
    PreconditionedConjugateGradientSolver,
    GeneralizedConjugateGradientSolver,
    ParallelKrylovFlowSolver,
    ParallelKrylovTransportSolver,
)
from imod.wq.vdf import VariableDensityFlow
from imod.wq.wel import Well
