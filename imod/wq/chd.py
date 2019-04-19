from imod.wq.pkgbase import BoundaryCondition


class ConstantHead(BoundaryCondition):
    _pkg_id = "chd"
    _mapping = (("shead", "head_start"), ("ehead", "head_end"))

    def __init__(self, head_start, head_end, concentration):
        super(__class__, self).__init__()
        self["head_start"] = head_start
        self["head_end"] = head_end
        self["concentration"] = concentration
