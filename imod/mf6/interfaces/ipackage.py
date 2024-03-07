import abc
from abc  import abstractmethod

from imod.mf6.interfaces.ipackagebase import IPackageBase
from typing import Any
import xarray as xr

class IPackage(IPackageBase, metaclass=abc.ABCMeta):
    """
    The base methods and attributes available in all packages
    """
    @abstractmethod
    def _valid(self, value):
        raise NotImplementedError        

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_non_grid_data(self, grid_names: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def auxiliary_data_fields(self) -> dict[str, str]:
        raise NotImplementedError