import abc
from typing import List

from imod.mf6.interfaces.ipackage import IPackage


class IMaskingSettings(IPackage, abc.ABC):
    """
    Interface for packages that support masking
    """

    @property
    @abc.abstractmethod
    def skip_variables(self) -> List[str]:
        raise NotImplementedError
