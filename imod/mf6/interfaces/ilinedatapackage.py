from abc import abstractmethod
from typing import TYPE_CHECKING

from imod.mf6.interfaces.ipackagebase import IPackageBase

if TYPE_CHECKING:
    import geopandas as gpd



class ILineDataPackage(IPackageBase):
    """
    Interface for packages for which the data is defined by lines independent of the domain definition.
    """

    @property
    @abstractmethod
    def line_data(self) -> "gpd.GeoDataFrame":
        raise NotImplementedError

    @line_data.setter
    @abstractmethod
    def line_data(self, value: "gpd.GeoDataFrame") -> None:
        raise NotImplementedError
