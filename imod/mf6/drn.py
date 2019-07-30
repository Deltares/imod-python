from imod.mf6.pgkbase import BoundaryCondition


class Drainage(BoundaryCondition):

    _pkg_id = "drn"

    _binary_data = ("elevation", "conductance", "observations")

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
        self["save_flows"] = save_flows
        self["print_input"] = print_input
        self["print_flows"] = print_flows
        self["save_flows"] = save_flows
        self["observations"] = observations
