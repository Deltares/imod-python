import abc
from abc import abstractmethod
from typing import Any

from imod.common.interfaces.ipackagebase import IPackageBase


class IPackage(IPackageBase, metaclass=abc.ABCMeta):
    """
    Interface for imod.mf6.package.Package
    """

    @abstractmethod
    def _valid(self, value):
        raise NotImplementedError

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_non_grid_data(self, grid_names: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def auxiliary_data_fields(self) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def _is_regridding_supported(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _is_grid_agnostic_package(self) -> bool:
        raise NotImplementedError
