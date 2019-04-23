from imod.wq.pkgbase import BoundaryCondition


class Drainage(BoundaryCondition):
    _pkg_id = "drn"

    _mapping = (("elevation", "elevation"), ("cond", "conductance"))

    def __init__(self, elevation, conductance, save_budget=False):
        super(__class__, self).__init__()
        self["elevation"] = elevation
        self["conductance"] = conductance
        self["save_budget"] = save_budget
