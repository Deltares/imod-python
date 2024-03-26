from imod.mf6.interfaces.idict import IDict
from imod.mf6.interfaces.imodel import IModel


class ISimulation(IDict):
    """
    Interface for imod.mf6.simulation.Modflow6Simulation
    """

    def is_split(self) -> bool:
        raise NotImplementedError

    def has_one_flow_model(self) -> bool:
        raise NotImplementedError

    def get_models(self) -> dict[str, IModel]:
        raise NotImplementedError
