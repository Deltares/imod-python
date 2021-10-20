from imod.wq.pkgbase import BoundaryCondition


class ConstantHead(BoundaryCondition):
    """
    The Constant Head package. The Time-Variant Specified-Head package is used
    to simulate specified head boundaries that can change within or between
    stress periods.

    Parameters
    ----------
    head_start: xr.DataArray of floats
        is the head at the boundary at the start of the stress period.
    head_end: xr.DataArray of floats
        is the head at the boundary at the end of the stress period.
    concentration: xr.DataArray of floats
        concentrations for the constant heads. It gets automatically written to
        the SSM package.
    save_budget: bool, optional
        is a flag indicating if the budget should be saved (ICHDCB).
        Default is False.
    """

    __slots__ = ("head_start", "head_end", "concentration", "save_budget")
    _pkg_id = "chd"
    _mapping = (("shead", "head_start"), ("ehead", "head_end"))

    def __init__(self, head_start, head_end, concentration, save_budget=False):
        super(__class__, self).__init__()
        self["head_start"] = head_start
        self["head_end"] = head_end
        self["concentration"] = concentration
        self["save_budget"] = save_budget

    def _pkgcheck(self, ibound=None):
        self._check_positive(["concentration"])
        self._check_location_consistent(["head_start", "head_end", "concentration"])

    def repeat_stress(self, head_start=None, head_end=None, use_cftime=False):
        varnames = ["head_start", "head_end"]
        values = [head_start, head_end]
        for varname, value in zip(varnames, values):
            self._repeat_stress(varname, value, use_cftime)
