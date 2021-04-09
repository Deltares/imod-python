from imod.flow.pkgbase import BoundaryCondition


class River(BoundaryCondition):
    """
    The River package is used to simulate head-dependent flux boundaries. In the
    River package if the head in the cell falls below a certain threshold, the
    flux from the river to the model cell is set to a specified lower bound.

    Parameters
    ----------
    stage: float or xr.DataArray of floats
        is the head in the river (STAGE), dims = ("layer", "y", "x").
    bottom_elevation: float or xr.DataArray of floats
        is the bottom of the riverbed (RBOT), dims = ("layer", "y", "x").
    conductance: float or xr.DataArray of floats
        is the conductance of the river, dims = ("layer", "y", "x").
    infiltration_factor: float or xr.DataArray of floats
        is the infiltration factor, dims = ("layer", "y", "x").
        This factor defines the reduces the conductance
        for infiltrating water compared to exfiltrating water:

        cond = A/(c * iff)

        where 'A' [L2] is the area where surface water
        and groundwater interact, 'c' [L] is the resistance,
        and 'iff' is the infiltration factor.

        The infiltration factor is thus equal or larger than 1.
    """

    _pkg_id = "riv"
    _variable_order = [
        "conductance",
        "stage",
        "bottom_elevation",
        "infiltration_factor",
    ]

    def __init__(
        self,
        conductance=None,
        stage=None,
        bottom_elevation=None,
        infiltration_factor=None,
    ):
        super(__class__, self).__init__()
        self.dataset["conductance"] = conductance
        self.dataset["stage"] = stage
        self.dataset["bottom_elevation"] = bottom_elevation
        self.dataset["infiltration_factor"] = infiltration_factor
