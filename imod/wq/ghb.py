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
    head: array of floats (xr.DataArray)
        head value for the GHB (BHEAD).
    conductance: array of floats (xr.DataArray)
        the conductance of the GHB (COND).
    concentration: "None" or array of floats (xr.DataArray), optional
        concentration of the GHB (CGHB), get automatically inserted into the SSM
        package.
        Default is "None".
    density: "None" or float, optional
        is the density used to convert the point head to the freshwater head
        (GHBSSMDENS). If not specified, the density is calculated dynamically
        using the concentration.
        Default is "None".
    save_budget: {True, False}, optional
        is a flag indicating if the budget should be saved (IGHBCB).
        Default is False.
    """

    _pkg_id = "ghb"
    _mapping = (("bhead", "head"), ("cond", "conductance"), ("ghbssmdens", "density"))

    def __init__(
        self, head, conductance, concentration=None, density=None, save_budget=False
    ):
        super(__class__, self).__init__()
        self["head"] = head
        self["conductance"] = conductance
        if concentration is not None:
            self["concentration"] = concentration
        if density is not None:
            self["density"] = density
        self["save_budget"] = save_budget

    def _pgkcheck(self, ibound=None):
        to_check = ["conductance"]
        if "concentration" in self.data_vars:
            to_check.append("concentration")
        if "density" in self.data_vars:
            to_check.append("density")
        self._check_positive(to_check)

        to_check.append("head")
        self._check_location_consistent(to_check)
