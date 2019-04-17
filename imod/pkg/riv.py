from imod.pkg.pkgbase import BoundaryCondition


class River(BoundaryCondition):
    _pkg_id = "riv"

    _mapping = (
        ("stage", "stage"),
        ("cond", "conductance"),
        ("rbot", "bottom_elevation"),
        ("rivssmdens", "density"),
    )

    def __init__(
        self,
        stage,
        conductance,
        bottom_elevation,
        concentration=None,
        density=None,
        save_budget=False,
    ):
        super(__class__, self).__init__()
        self["stage"] = stage
        self["conductance"] = conductance
        self["bottom_elevation"] = bottom_elevation
        if concentration is not None:
            self["concentration"] = concentration
        if density is not None:
            self["density"] = density
        self["save_budget"] = save_budget
