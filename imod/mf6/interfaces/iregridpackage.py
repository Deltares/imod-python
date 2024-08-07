import abc
from typing import Optional

from imod.mf6.interfaces.ipackage import IPackage
from imod.mf6.regrid.regrid_schemes import RegridMethodType


class IRegridPackage(IPackage, abc.ABC):
    """
    Interface for packages that support regridding
    """

    @abc.abstractmethod
    def get_regrid_methods(self) -> Optional[RegridMethodType]:
        raise NotImplementedError
