from imod.couplers.metamod.node_svat_mapping import NodeSvatMapping
from imod.mf6 import Modflow6Simulation
from imod.msw import MetaSwapModel


class MetaMod:
    """[summary]

    Parameters
    ----------
    msw_model : MetaSwapModel
        The MetaSWAP model that should be coupled.
    mf6_simulation : Modflow6Simulation
        The Modflow6 simulation that should be coupled.
    """

    def __init__(self, msw_model: MetaSwapModel, mf6_simulation: Modflow6Simulation):
        self.msw_model = msw_model
        self.mf6_simulation = mf6_simulation

    def write(self, directory):
        self.msw_model.write(directory)
        self.mf6_simulation.write(directory)
        self.write_exchanges(directory)

    def write_exchanges(self, directory):
        area = self.msw_model["grid_data"].dataset["area"]
        active = self.msw_model["grid_data"].dataset["active"]
        grid_mapping = NodeSvatMapping(area, active)
        grid_mapping.write(directory)
