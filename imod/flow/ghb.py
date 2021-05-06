from imod.flow.pkgbase import BoundaryCondition


class GeneralHeadBoundary(BoundaryCondition):
    """
    The General-Head Boundary package is used to simulate head-dependent flux
    boundaries. In the General-Head Boundary package the flux is always
    proportional to the difference in head.

    Parameters
    ----------
    head: float or xr.DataArray of floats
        head value for the GHB (BHEAD), dims ``("layer", "y", "x")``.
    conductance: float or xr.DataArray of floats
        the conductance of the GHB (COND), dims ``("layer", "y", "x")``.
    """

    _pkg_id = "ghb"
    _variable_order = ["conductance", "head"]

    def __init__(self, conductance, head):
        super(__class__, self).__init__()
        self.dataset["conductance"] = conductance
        self.dataset["head"] = head
