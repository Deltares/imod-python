from abc import abstractmethod

from numpy import ndarray

from imod.mf6.interfaces.ipackagebase import IPackageBase


class ILineDataPackage(IPackageBase):
    """
    Interface for packages for which the data is defined by lines independent of the domain definition.
    """

    @property
    @abstractmethod
    def geometry(self) -> ndarray[object]:
        raise NotImplementedError
