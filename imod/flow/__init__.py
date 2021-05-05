"""
Create iMODFLOW model.

With this module full iMODFLOW models can be written to disk. 
Models in :mod:`imod.flow` are created in a similar way to those created in :mod:`imod.wq`.
The ImodflowModel object by default writes iMODFLOW projectfiles now 
instead of the older iMODFLOW runfiles.
"""

from imod.flow.ani import HorizontalAnisotropy
from imod.flow.bas import Boundary, Top, Bottom, StartingHead
from imod.flow.conductivity import (
    HorizontalHydraulicConductivity,
    VerticalHydraulicConductivity,
    VerticalAnisotropy,
)
from imod.flow.riv import River
from imod.flow.cap import MetaSwap
from imod.flow.drn import Drain
from imod.flow.ghb import GeneralHeadBoundary
from imod.flow.wel import Well
from imod.flow.chd import ConstantHead
from imod.flow.model import ImodflowModel
from imod.flow.dis import TimeDiscretization
from imod.flow.slv import PreconditionedConjugateGradientSolver
from imod.flow.sto import StorageCoefficient, SpecificStorage
from imod.flow.hfb import HorizontalFlowBarrier
from imod.flow.rch import Recharge
from imod.flow.evt import EvapoTranspiration


def write(path, model, name=None, runfile_parameters=None, output_packages=["shd"]):
    """Deprectated function"""
    raise DeprecationWarning(
        "This function has been deprecated. Use imod.flow.ImodflowModel instead."
    )
