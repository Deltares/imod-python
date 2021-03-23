from imod.flow.pkgbase import BoundaryCondition

class River(BoundaryCondition):
    """
    The River package is used to simulate head-dependent flux boundaries. In the
    River package if the head in the cell falls below a certain threshold, the
    flux from the river to the model cell is set to a specified lower bound.

    Parameters
    ----------
    stage: float or xr.DataArray of floats
        is the head in the river (STAGE).
    bottom_elevation: float or xr.DataArray of floats
        is the bottom of the riverbed (RBOT).
    conductance: float or xr.DataArray of floats
        is the conductance of the river.
    infiltration_factor: float or xr.DataArray of floats
        is the infiltration factor. This factor defines the 
        extra resistance exerted for infiltrating water compared to
        exfiltrating water.
    """

    _pkg_id = "riv"
    _variable_order = [
        "conductance", 
        "stage", 
        "bottom_elevation", 
        "infiltration_factor"
        ]

    def __init__(
        self, conductance=None, stage=None, 
        bottom_elevation=None, infiltration_factor=None
        ):
        super(__class__, self).__init__()
        self.dataset["conductance"] = conductance
        self.dataset["stage"] = stage
        self.dataset["bottom_elevation"] = bottom_elevation
        self.dataset["infiltration_factor"] = infiltration_factor
    
    