from imod.pkg.pkgbase import BoundaryCondition


class Drainage(BoundaryCondition):
    _pkg_id = "drn"

    _mapping = (("elevation", "elevation"), ("cond", "conductance"))

    def __init__(self, elevation, conductance):
        super(__class__, self).__init__()
        self["elevation"] = elevation
        self["conductance"] = conductance
