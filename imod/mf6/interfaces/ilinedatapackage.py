from abc import abstractmethod

import geopandas as gpd

from imod.mf6.interfaces.ipackagebase import IPackageBase


class ILineDataPackage(IPackageBase):
    """
    Interface for packages for which the data is defined by lines independent of the domain definition.
    """

    @property
    @abstractmethod
    def geometry(self) -> gpd.GeoDataFrame:
        raise NotImplementedError

    @geometry.setter
    @abstractmethod
    def geometry(self, value: gpd.GeoDataFrame) -> None:
        raise NotImplementedError
