from imod.flow.pkgbase import BoundaryCondition


class Drain(BoundaryCondition):
    """
    The Drain package is used to simulate head-dependent flux boundaries. In
    the Drain package if the head in the cell falls below a certain threshold,
    the flux from the drain to the model cell drops to zero.

    Parameters
    ----------
    elevation: float or xr.DataArray of floats
        elevation of the drain, dims ``("layer", "y", "x")``.
    conductance: float or xr.DataArray of floats
        is the conductance of the drain, dims ``("layer", "y", "x")``.
    """

    _pkg_id = "drn"
    _variable_order = ["conductance", "elevation"]

    def __init__(self, conductance=None, elevation=None):
        super(__class__, self).__init__()
        self.dataset["conductance"] = conductance
        self.dataset["elevation"] = elevation
