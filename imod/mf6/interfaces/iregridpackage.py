
from imod.mf6.interfaces.ipackage import IPackage
import abc
from typing import Optional
from imod.mf6.utilities.regridding_types import RegridderType

class IRegridPackage(IPackage, abc.ABC):
    @property
    def regrid_methods(self) -> Optional[dict[str, tuple[RegridderType, str]]]:
        if hasattr(self, "_regrid_method"):
            return self._regrid_method
        return None