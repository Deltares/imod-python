from dataclasses import dataclass

from imod.prepare.topsystem.allocation import ALLOCATION_OPTION
from imod.prepare.topsystem.conductance import DISTRIBUTING_OPTION


@dataclass()
class SimulationAllocationOptions:
    """
    Object containing allocation otpions, specified per packages type on
    importing fron imod5. Can be used to set defaults when importing a
    simulation or a GroundwaterFlowModel from imod5.

    Parameters
    ----------
    drn: allocation option to be used for drainage packages
    riv: allocation option to be used for river packages

    """

    drn: ALLOCATION_OPTION = ALLOCATION_OPTION.first_active_to_elevation
    riv: ALLOCATION_OPTION = ALLOCATION_OPTION.stage_to_riv_bot
    ghb: ALLOCATION_OPTION = ALLOCATION_OPTION.at_elevation


@dataclass()
class SimulationDistributingOptions:
    """
    Object containing conductivity distribution methods, specified per packages
    type. Can be used to set defaults when importing a simulation or a
    GroundwaterFlowModel from imod5.

    Parameters
    ----------
    drn: distribution option to be used for drainage packages
    riv: distribution option to be used for river packages

    """

    drn: DISTRIBUTING_OPTION = DISTRIBUTING_OPTION.by_corrected_transmissivity
    riv: DISTRIBUTING_OPTION = DISTRIBUTING_OPTION.by_corrected_transmissivity
    ghb: DISTRIBUTING_OPTION = DISTRIBUTING_OPTION.by_layer_transmissivity
