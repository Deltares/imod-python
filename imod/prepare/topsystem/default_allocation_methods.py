from dataclasses import dataclass

from imod.prepare.topsystem.allocation import ALLOCATION_OPTION
from imod.prepare.topsystem.conductance import DISTRIBUTING_OPTION


@dataclass()
class SimulationAllocationOptions:
    """
    Object containing default allocation otpions for different packages
    on importing fron imod5.

    Parameters
    ----------
    drn: allocation option to be used for drainage packages
    riv: allocation option to be used for river packages

    """

    drn: ALLOCATION_OPTION = ALLOCATION_OPTION.first_active_to_elevation
    riv: ALLOCATION_OPTION = ALLOCATION_OPTION.stage_to_riv_bot


@dataclass()
class SimulationDistributingOptions:
    """
    Object containing  conductivity distribution methods for different packages
    on importing fron imod5.

    Parameters
    ----------
    drn: distribution option to be used for drainage packages
    riv: distribution option to be used for river packages

    """

    drn: DISTRIBUTING_OPTION = DISTRIBUTING_OPTION.by_corrected_transmissivity
    riv: DISTRIBUTING_OPTION = DISTRIBUTING_OPTION.by_corrected_transmissivity
