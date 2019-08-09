from imod.mf6.pkgbase import BoundaryCondition


class Well(BoundaryCondition):
    _pkg_id = "wel"

    def __init__(
        self,
        layer,
        row,
        column,
        rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        super(__class__, self).__init__()
        self["layer"] = layer
        self["row"] = row
        self["column"] = column
        self["print_input"] = print_input
        self["print_flows"] = print_flows
        self["save_flows"] = save_flows
        self["observations"] = observations
        self._initialize_template()
