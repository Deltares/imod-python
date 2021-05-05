from imod.flow.pkgbase import TopBoundaryCondition


class Recharge(TopBoundaryCondition):
    """
    Recharge provides a fixed flux boundary condition to the top layer of the
    groundwater system.  Note that unlike in iMOD-WQ, there is only the option
    in iMODFLOW to apply the recharge package to the top layer.

    Parameters
    ----------
    rate: float or xr.DataArray of floats
        recharge rate in mm/day (NOTA BENE!), dims ``("time", "y", "x")``.
    """

    _pkg_id = "rch"
    _variable_order = ["rate"]

    def __init__(self, rate):
        super(__class__, self).__init__()
        self.dataset["rate"] = rate
