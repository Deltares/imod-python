from abc import ABC, abstractmethod

import xarray as xr


class IPackageBase(ABC):
    """
    The base methods and attributes available in all packages
    """

    @property
    @abstractmethod
    def dataset(self) -> xr.Dataset:
        raise NotImplementedError

    @dataset.setter
    @abstractmethod
    def dataset(self, value: xr.Dataset) -> None:
        raise NotImplementedError
