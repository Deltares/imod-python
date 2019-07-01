from imod.wq.pkgbase import BoundaryCondition


class Drainage(BoundaryCondition):
    """
    The Drain package is used to simulate head-dependent flux boundaries. In the
    Drain package if the head in the cell falls below a certain threshold, the
    flux from the drain to the model cell drops to zero.

    Parameters
    ----------
    elevation: array of floats (xr.DataArray)
        elevation of the drain.
    conductance: array of floats (xr.DataArray)
        is the conductance of the drain.
    save_budget: {True, False}, optional
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If save_budget is True cell-by-cell budget data will be saved.
        Default is False.
    """

    _pkg_id = "drn"

    _mapping = (("elevation", "elevation"), ("cond", "conductance"))

    def __init__(self, elevation, conductance, save_budget=False):
        super(__class__, self).__init__()
        self["elevation"] = elevation
        self["conductance"] = conductance
        self["save_budget"] = save_budget

    def _pkgcheck(self, ibound=None):
        self._check_positive(["conductance"])
        self._check_location_consistent(["elevation", "conductance"])
