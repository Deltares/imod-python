"""
Create iMODFLOW model.

With this module full iMODFLOW models can be written to disk. 
Models in :mod:`imod.flow` are created in a similar way to those created in :mod:`imod.wq`.
The ImodflowModel object by default writes iMODFLOW projectfiles now 
instead of the older iMODFLOW runfiles.
"""

from imod.flow.bas import Boundary, Top, Bottom, StartingHead
from imod.flow.conductivity import (
    HorizontalHydraulicConductivity,
    VerticalHydraulicConductivity,
    VerticalAnistropy,
)
from imod.flow.riv import River
from imod.flow.drn import Drain
from imod.flow.ghb import GeneralHeadBoundary
from imod.flow.wel import Well
from imod.flow.chd import ConstantHead
from imod.flow.model import ImodflowModel
from imod.flow.dis import TimeDiscretization
from imod.flow.slv import PreconditionedConjugateGradientSolver
from imod.flow.sto import StorageCoefficient

def write(path, model, name=None, runfile_parameters=None, output_packages=["shd"]):
    """Deprectated function"""
    raise DeprecationWarning(
        "This function has been deprecated. " 
        "Instead initiate an imod.flow.ImodflowModel instance and use its' write method."
        )
