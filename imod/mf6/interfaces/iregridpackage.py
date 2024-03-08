
import abc
from typing import Optional

from imod.mf6.interfaces.ipackage import IPackage
from imod.mf6.utilities.regridding_types import RegridderType


class IRegridPackage(IPackage, abc.ABC):
    """
    Interface for packages that support regridding
    """
    @property
    def regrid_methods(self) -> Optional[dict[str, tuple[RegridderType, str]]]:
        if hasattr(self, "_regrid_method"):
            return self._regrid_method
        return None