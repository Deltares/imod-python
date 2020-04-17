"""
Create a structured Modflow 6 model.
"""

from imod.mf6 import multimodel
from imod.mf6.chd import ConstantHead
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.drn import Drainage
from imod.mf6.evt import Evapotranspiration
from imod.mf6.ghb import GeneralHeadBoundary
from imod.mf6.ic import InitialConditions
from imod.mf6.ims import (Solution, SolutionPresetComplex,
                          SolutionPresetModerate, SolutionPresetSimple)
from imod.mf6.model import GroundwaterFlowModel
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.oc import OutputControl
from imod.mf6.out import open_hds
from imod.mf6.rch import Recharge
from imod.mf6.riv import River
from imod.mf6.simulation import Modflow6Simulation
from imod.mf6.sto import Storage
from imod.mf6.timedis import TimeDiscretization
from imod.mf6.wel import Well
