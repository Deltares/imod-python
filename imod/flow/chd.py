from imod.flow.pkgbase import BoundaryCondition


class ConstantHead(BoundaryCondition):
    """
    The Constant Head package. The Time-Variant Specified-Head package is used
    to simulate specified head boundaries that can change between
    stress periods.

    Parameters
    ----------
    head: xr.DataArray of floats
        is the head at the boundary
    """

    _pkg_id = "chd"
    _variable_order = ["head"]

    def __init__(self, head):
        super().__init__()
        self.dataset["head"] = head
