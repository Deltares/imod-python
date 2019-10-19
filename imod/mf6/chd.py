from imod.mf6.pkgbase import BoundaryCondition


class ConstantHead(BoundaryCondition):
    __slots__ = ("head", "print_input", "print_flows", "save_flows", "observations")
    _pkg_id = "chd"
    _binary_data = ("head",)
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(
        self,
        head,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        super(__class__, self).__init__()
        self["head"] = head
        self["print_input"] = print_input
        self["print_flows"] = print_flows
        self["save_flows"] = save_flows
        self["observations"] = observations
