from imod.mf6.pkgbase import BoundaryCondition


class Drainage(BoundaryCondition):

    __slots__ = (
        "elevation",
        "conductance",
        "print_input",
        "print_flows",
        "save_flows",
        "observations",
    )
    _pkg_id = "drn"
    # has to be ordered as in the list
    _binary_data = ("elevation", "conductance")
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(
        self,
        elevation,
        conductance,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        super(__class__, self).__init__()
        self["elevation"] = elevation
        self["conductance"] = conductance
        self["print_input"] = print_input
        self["print_flows"] = print_flows
        self["save_flows"] = save_flows
        self["observations"] = observations
