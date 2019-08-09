"""
Create a structured Modflow 6 model.
"""

# make classes directly available under imod.mf6
from imod.mf6.drn import Drainage
from imod.mf6.timedis import TimeDiscretization
from imod.mf6.pkgbase import Package, BoundaryCondition
from imod.mf6.dis import StructuredDiscretization
