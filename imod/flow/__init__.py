"""
Create iMODFLOW model.

With this module full iMODFLOW models can be written to disk. Models in
:mod:`imod.flow` are created in a similar way to those created in
:mod:`imod.wq`.  The ImodflowModel object by default writes iMODFLOW
projectfiles now instead of the older iMODFLOW runfiles.
"""

from imod.flow.ani import HorizontalAnisotropy
from imod.flow.bas import Bottom, Boundary, StartingHead, Top
from imod.flow.cap import MetaSwap
from imod.flow.chd import ConstantHead
from imod.flow.conductivity import (
    HorizontalHydraulicConductivity,
    Transmissivity,
    VerticalAnisotropy,
    VerticalHydraulicConductivity,
    VerticalResistance,
)
from imod.flow.dis import TimeDiscretization
from imod.flow.drn import Drain
from imod.flow.evt import EvapoTranspiration
from imod.flow.ghb import GeneralHeadBoundary
from imod.flow.hfb import HorizontalFlowBarrier
from imod.flow.model import ImodflowModel
from imod.flow.oc import OutputControl
from imod.flow.rch import Recharge
from imod.flow.riv import River
from imod.flow.slv import PreconditionedConjugateGradientSolver
from imod.flow.sto import SpecificStorage, StorageCoefficient
from imod.flow.wel import Well


def write(path, model, name=None, runfile_parameters=None, output_packages=["shd"]):
    """Removed function"""
    raise NotImplementedError(
        "This function has been removed. Use imod.flow.ImodflowModel instead."
    )
