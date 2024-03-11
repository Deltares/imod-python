
import abc
from typing import Optional

from imod.mf6.interfaces.ipackage import IPackage
from imod.mf6.utilities.regridding_types import RegridderType


class IRegridPackage(IPackage, abc.ABC):
    """
    Interface for packages that support regridding
    """
    pass