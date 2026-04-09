from abc import abstractmethod

from imod.common.interfaces.idict import IDict
from imod.common.interfaces.imodel import IModel


class ISimulation(IDict):
    """
    Interface for imod.mf6.simulation.Modflow6Simulation
    """

    @abstractmethod
    def is_split(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _has_one_flow_model(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_models(self) -> dict[str, IModel]:
        raise NotImplementedError

    @abstractmethod
    def get_models_of_type(self, model_id: str) -> dict[str, IModel]:
        raise NotImplementedError
