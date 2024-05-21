from abc import abstractmethod

from imod.mf6.interfaces.idict import IDict
from imod.mf6.interfaces.imodel import IModel


class ISimulation(IDict):
    """
    Interface for imod.mf6.simulation.Modflow6Simulation
    """

    @abstractmethod
    def is_split(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def has_one_flow_model(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_models(self) -> dict[str, IModel]:
        raise NotImplementedError

    @abstractmethod
    def get_models_of_type(self, model_id: str) -> dict[str, IModel]:
        raise NotImplementedError
