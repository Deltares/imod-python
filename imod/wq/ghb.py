from imod.wq.pkgbase import BoundaryCondition

# class GeneralHeadBoundaryGroup(object):
# Does a groupby over packages of the same kind when writing
# Collects total data of all same kind packages
# adds a system number
# This one is actually in charge of generating the output from
# the dictionaries provided by the ._compose_values methods
# Every system is treated independently


class GeneralHeadBoundary(BoundaryCondition):
    """
    The General-Head Boundary package is used to simulate head-dependent flux
    boundaries. In the General-Head Boundary package the flux is always
    proportional to the difference in head.

    Parameters
    ----------
    head: float or xr.DataArray of floats
        head value for the GHB (BHEAD).
    conductance: float or xr.DataArray of floats
        the conductance of the GHB (COND).
    density: float or xr.DataArray of floats
        is the density used to convert the point head to the freshwater head
        (GHBSSMDENS).
    concentration: "None" or xr.DataArray of floats, optional
        concentration of the GHB (CGHB), get automatically inserted into the SSM
        package.
        Default is "None".
    save_budget: bool, optional
        is a flag indicating if the budget should be saved (IGHBCB).
        Default is False.
    """

    __slots__ = ("head", "conductance", "density", "concentration", "save_budget")
    _pkg_id = "ghb"
    _mapping = (("bhead", "head"), ("cond", "conductance"), ("ghbssmdens", "density"))

    def __init__(
        self, head, conductance, density, concentration=None, save_budget=False
    ):
        super(__class__, self).__init__()
        self["head"] = head
        self["conductance"] = conductance
        self["density"] = density
        if concentration is not None:
            self["concentration"] = concentration
        self["save_budget"] = save_budget

    def _pkgcheck(self, ibound=None):
        to_check = ["conductance", "density"]
        if "concentration" in self.data_vars:
            to_check.append("concentration")
        self._check_positive(to_check)

        to_check.append("head")
        self._check_location_consistent(to_check)

    def repeat_stress(
        self,
        head=None,
        conductance=None,
        density=None,
        concentration=None,
        use_cftime=False,
    ):
        varnames = ["head", "conductance", "density", "concentration"]
        values = [head, conductance, density, concentration]
        for varname, value in zip(varnames, values):
            self._repeat_stress(varname, value, use_cftime)
