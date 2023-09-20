from abc import abstractmethod

from numpy import ndarray

from imod.mf6.interfaces.ipackagebase import IPackageBase


class IPointDataPackage(IPackageBase):
    """
    Interface for packages for which the data is defined by x and y coordinates independent of the domain definition.
    """

    @property
    @abstractmethod
    def x(self) -> ndarray[float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def y(self) -> ndarray[float]:
        raise NotImplementedError
