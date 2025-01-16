from dataclasses import dataclass

from imod.prepare.topsystem.allocation import ALLOCATION_OPTION
from imod.prepare.topsystem.conductance import DISTRIBUTING_OPTION


@dataclass()
class SimulationAllocationOptions:
    """
    Object containing default allocation options, specified per packages type on
    importing from imod5. Can be used to set defaults when importing a
    simulation or a GroundwaterFlowModel from imod5.

    Parameters
    ----------
    drn: ALLOCATION_OPTION
        allocation option to be used for drainage packages, defaults to
        ``first_active_to_elevation``.
    riv: ALLOCATION_OPTION
        allocation option to be used for river packages, defaults to
        ``stage_to_riv_bot_drn_above``.
    ghb: ALLOCATION_OPTION
        allocation option to be used for general head boundary packages,
        defaults to ``at_elevation``.

    Examples
    --------

    Initiate allocation default options

    >>> alloc_options = SimulationAllocationOptions()

    You can set different options as follows:

    >>> from imod.prepare.topsystem import ALLOCATION_OPTION
    >>> alloc_options.riv = ALLOCATION_OPTION.at_elevation

    """

    drn: ALLOCATION_OPTION = ALLOCATION_OPTION.first_active_to_elevation
    riv: ALLOCATION_OPTION = ALLOCATION_OPTION.stage_to_riv_bot_drn_above
    ghb: ALLOCATION_OPTION = ALLOCATION_OPTION.at_elevation


@dataclass()
class SimulationDistributingOptions:
    """
    Object containing conductance distribution methods, specified per packages
    type. Can be used to set defaults when importing a simulation or a
    GroundwaterFlowModel from imod5.

    Parameters
    ----------
    drn: DISTRIBUTING_OPTION
        distribution option to be used for drainage packages, defaults to
        ``by_corrected_transmissivity``.
    riv: DISTRIBUTING_OPTION
        distribution option to be used for river packages, defaults to
        ``by_corrected_transmissivity``.
    ghb: DISTRIBUTING_OPTION
        distribution option to be used for general head boundary packages,
        defaults to ``by_layer_transmissivity``.

    Examples
    --------

    Initiate default distributing options

    >>> dist_options = SimulationDistributingOptions()

    You can set different options as follows:

    >>> from imod.prepare.topsystem import DISTRIBUTING_OPTION
    >>> dist_options.riv = DISTRIBUTING_OPTION.by_layer_transmissivity

    """

    drn: DISTRIBUTING_OPTION = DISTRIBUTING_OPTION.by_corrected_transmissivity
    riv: DISTRIBUTING_OPTION = DISTRIBUTING_OPTION.by_corrected_transmissivity
    ghb: DISTRIBUTING_OPTION = DISTRIBUTING_OPTION.by_layer_transmissivity
