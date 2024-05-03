from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from imod.mf6.interfaces.ipackagebase import IPackageBase


class IPointDataPackage(IPackageBase):
    """
    Interface for packages for which the data is defined by x and y coordinates independent of the domain definition.
    """

    @property
    @abstractmethod
    def x(self) -> NDArray[np.float64]:
        raise NotImplementedError

    @property
    @abstractmethod
    def y(self) -> NDArray[np.float64]:
        raise NotImplementedError
