from imod.mf6.pkgbase import BoundaryCondition


class River(BoundaryCondition):
    __slots__ = (
        "stage",
        "conductance",
        "bottom_elevation",
        "print_input",
        "print_flows",
        "save_flows",
        "observations",
    )
    _pkg_id = "riv"
    _binary_data = ("stage", "conductance", "bottom_elevation")
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(
        self,
        stage,
        conductance,
        bottom_elevation,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        super(__class__, self).__init__()
        self["stage"] = stage
        self["conductance"] = conductance
        self["bottom_elevation"] = bottom_elevation
        self["print_input"] = print_input
        self["print_flows"] = print_flows
        self["save_flows"] = save_flows
        self["observations"] = observations
