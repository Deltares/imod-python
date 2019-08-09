from imod.mf6.pkgbase import BoundaryCondition


class Recharge(BoundaryCondition):
    _pkg_id = "rch"

    def __init__(
        self,
        rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        self["rate"] = rate
        self["print_input"] = print_input
        self["print_flows"] = print_flows
        self["save_flows"] = save_flows
        self["observations"] = observations
        self._initialize_template()
