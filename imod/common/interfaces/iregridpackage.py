import abc
from typing import Optional

from imod.common.interfaces.ipackagebase import IPackageBase
from imod.common.utilities.dataclass_type import DataclassType


class IRegridPackage(IPackageBase, abc.ABC):
    """
    Interface for packages that support regridding
    """

    @abc.abstractmethod
    def get_regrid_methods(self) -> Optional[DataclassType]:
        raise NotImplementedError
