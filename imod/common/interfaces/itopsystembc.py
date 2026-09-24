from abc import abstractmethod

from imod.common.interfaces.ipackage import IPackage
from imod.typing import GridDataDict, GridDataset


class ITopSystemBoundaryCondition(IPackage):
    """
    Interface for top system boundary condition packages in MODFLOW 6.
    """

    @classmethod
    @abstractmethod
    def aggregate_layers(cls, dataset: GridDataset) -> GridDataDict:
        raise NotImplementedError
