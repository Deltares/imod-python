from abc import abstractmethod

from imod.common.interfaces.ipackagebase import IPackageBase
from imod.typing import GridDataArray


class IAgnosticPackage(IPackageBase):
    """
    Interface for packages for which the data is defined independent of the domain definition.
    """

    @abstractmethod
    def to_mf6_pkg(
        self,
        idomain: GridDataArray,
        top: GridDataArray,
        bottom: GridDataArray,
        k: GridDataArray,
        validate: bool = False,
        strict_validation: bool = True,
    ) -> IPackageBase:
        raise NotImplementedError
