from imod.couplers.metamod.node_svat_mapping import NodeSvatMapping
from imod.couplers.metamod.rch_svat_mapping import RechargeSvatMapping
from imod.mf6 import Modflow6Simulation
from imod.msw import GridData, MetaSwapModel


class MetaMod:
    """
    The MetaMod class creates the necessary input files for coupling MetaSWAP to MODFLOW 6.

    Parameters
    ----------
    msw_model : MetaSwapModel
        The MetaSWAP model that should be coupled.
    mf6_simulation : Modflow6Simulation
        The Modflow6 simulation that should be coupled.
    """

    # TODO:
    # - NodeSvatMapping: check with idomain if no coupling is done to inactive
    #   cells.
    # - rchindex2svat.dxc
    # - wellindex2svat.dxc
    # - a toml file

    def __init__(self, msw_model: MetaSwapModel, mf6_simulation: Modflow6Simulation):
        self.msw_model = msw_model
        self.mf6_simulation = mf6_simulation

    def write(self, directory):
        # For some reason the Modflow 6 model has to be written first, before
        # writing the MetaSWAP model. Else we get an Access Violation Error when
        # running the coupler.
        self.mf6_simulation.write(directory / "Modflow6")
        self.msw_model.write(directory / "MetaSWAP")
        # Exchange files should be in Modflow 6 model directory
        self.write_exchanges(directory / "Modflow6")

    def write_exchanges(self, directory):
        grid_data_key = [
            pkgname
            for pkgname, pkg in self.msw_model.items()
            if isinstance(pkg, GridData)
        ][0]

        area = self.msw_model[grid_data_key].dataset["area"]
        active = self.msw_model[grid_data_key].dataset["active"]
        grid_mapping = NodeSvatMapping(area, active)
        grid_mapping.write(directory)

        gwf_names = [
            key
            for key, value in self.mf6_simulation.items()
            if value._pkg_id == "model"
        ]

        # Assume only one groundwater flow model
        # FUTURE: Support multiple groundwater flow models.
        gwf_model = self.mf6_simulation[gwf_names[0]]

        if "rch_msw" not in gwf_model.keys():
            raise ValueError(
                "No package named 'rch_msw' detected in Modflow 6 model. "
                "iMOD_coupler requires a Recharge package with 'rch_msw' as name"
            )

        recharge = gwf_model["rch_msw"]

        rch_mapping = RechargeSvatMapping(area, active, recharge)
        rch_mapping.write(directory)
