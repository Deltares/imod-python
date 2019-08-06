from imod.wq.pkgbase import BoundaryCondition


class ConstantHead(BoundaryCondition):
    """
    The Constant Head package. The Time-Variant Specified-Head package is used
    to simulate specified head boundaries that can change within or between
    stress periods.

    Parameters
    ----------
    head_start: array of floats (xr.DataArray)
        is the head at the boundary at the start of the stress period.
    head_end: array of floats (xr.DataArray)
        is the head at the boundary at the end of the stress period.
    concentration: array of floats (xr.DataArray)
        concentrations for the constant heads. It gets automatically written to
        the SSM package.
    """

    _pkg_id = "chd"
    _mapping = (("shead", "head_start"), ("ehead", "head_end"))

    def __init__(self, head_start, head_end, concentration):
        super(__class__, self).__init__()
        self["head_start"] = head_start
        self["head_end"] = head_end
        self["concentration"] = concentration

    def _pkgcheck(self, ibound=None):
        self._check_positive(["concentration"])
        self._check_location_consistent(["head_start", "head_end", "concentration"])

    def add_timemap(self, head_start=None, head_end=None, use_cftime=False):
        varnames = ["head_start", "head_end"]
        values = [head_start, head_end]
        for varname, value in zip(varnames, values):
            self._add_timemap(varname, value, use_cftime)
