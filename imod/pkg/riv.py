from imod.pkg.pkgbase import BoundaryCondition


class River(BoundaryCondition):
    _pkg_id = "riv"

    _mapping = (
        ("stage", "stage"),
        ("cond", "conductance"),
        ("rbot", "bottom_elevation"),
        ("rivssmdens", "density"),
    )

    def __init__(self, stage, conductance, bottom_elevation, concentration, density, save_budget=False):
        super(__class__, self).__init__()
        self["stage"] = stage
        self["conductance"] = conductance
        self["bottom_elevation"] = bottom_elevation
        self["concentration"] = concentration
        self["density"] = density
        self["save_budget"] = save_budget
