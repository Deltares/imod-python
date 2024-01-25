from imod.mf6.interfaces.ipackagebase import IPackageBase

import abc
from typing import Dict
class IPackage(IPackageBase, metaclass=abc.ABCMeta):
    


    """
    The base methods and attributes available in all packages
    """

    dataset: None

    @property
    @abc.abstractmethod
    def auxiliary_data_fields(self) -> Dict[str, str]:
        raise NotImplementedError