from imod.couplers.metamod.node_svat_mapping import NodeSvatMapping


class MetaMod:
    def __init__(self, msw_model, mf6_simulation):
        self.msw_model = msw_model
        self.mf6_simulation = mf6_simulation

    def write(self, directory):
        self.msw_model.write(directory)
        self.mf6_simulation.write(directory)
        self.write_exchanges(directory)

    def write_exchanges(self, directory):
        # Grid mapping
        area = self.msw_model["grid_data"].dataset["area"]
        active = self.msw_model["grid_data"].dataset["active"]
        grid_mapping = NodeSvatMapping(area, active)
        grid_mapping.write(directory)
