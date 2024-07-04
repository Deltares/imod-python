from abc import ABC, abstractmethod

import xarray as xr

from imod.typing import GridDataset


class IPackageBase(ABC):
    """
    Interface for imod.mf6.pkgbase.PackageBase
    """

    @property
    @abstractmethod
    def dataset(self) -> xr.Dataset:
        raise NotImplementedError

    @dataset.setter
    @abstractmethod
    def dataset(self, value: xr.Dataset) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _from_dataset(self, ds: GridDataset):
        raise NotImplementedError
