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
