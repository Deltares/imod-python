from imod.mf6.pkgbase import BoundaryCondition


class Recharge(BoundaryCondition):
    __slots__ = ("rate", "print_input", "print_flows", "save_flows", "observations")
    _pkg_id = "rch"
    _binary_data = ("rate",)

    def __init__(
        self,
        rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        super(__class__, self).__init__()
        self["rate"] = rate
        self["print_input"] = print_input
        self["print_flows"] = print_flows
        self["save_flows"] = save_flows
        self["observations"] = observations
        self._initialize_template()
