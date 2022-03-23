import warnings

from imod.couplers.metamod.node_svat_mapping import NodeSvatMapping
from imod.couplers.metamod.rch_svat_mapping import RechargeSvatMapping
from imod.couplers.metamod.wel_svat_mapping import WellSvatMapping
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
        gwf_names = [
            key
            for key, value in self.mf6_simulation.items()
            if value._pkg_id == "model"
        ]
        # Assume only one groundwater flow model
        # FUTURE: Support multiple groundwater flow models.
        gwf_model = self.mf6_simulation[gwf_names[0]]

        grid_data_key = [
            pkgname
            for pkgname, pkg in self.msw_model.items()
            if isinstance(pkg, GridData)
        ][0]

        dis = gwf_model[gwf_model._get_pkgkey("dis")]

        index, svat = self.msw_model[grid_data_key].generate_index_array()
        grid_mapping = NodeSvatMapping(svat, dis)
        grid_mapping.write(directory, index, svat)

        if "rch_msw" not in gwf_model.keys():
            raise ValueError(
                "No package named 'rch_msw' detected in Modflow 6 model. "
                "iMOD_coupler requires a Recharge package with 'rch_msw' as name"
            )

        recharge = gwf_model["rch_msw"]

        rch_mapping = RechargeSvatMapping(svat, recharge)
        rch_mapping.write(directory, index, svat)

        if "wells_msw" in gwf_model.keys():
            well = gwf_model["wells_msw"]

            well_mapping = WellSvatMapping(svat, well)
            well_mapping.write(directory, index, svat)
        else:
            warnings.warn(
                "No package named 'wells_msw' detected, "
                "no wells are coupled to MetaSWAP",
                UserWarning,
            )
