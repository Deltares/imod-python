import abc
from typing import Optional

from imod.common.interfaces.ipackage import IPackage
from imod.common.utilities.dataclass_type import DataclassType


class IRegridPackage(IPackage, abc.ABC):
    """
    Interface for packages that support regridding
    """

    @abc.abstractmethod
    def get_regrid_methods(self) -> Optional[DataclassType]:
        raise NotImplementedError
