from imod.mf6.pkgbase import BoundaryCondition


class Evapotranspiration(BoundaryCondition):
    _pkg_id = "evt"
    _binary_data = ("surface", "rate", "depth", "proportion_depth", "proportion_rate")

    def __init__(
        self,
        surface,
        rate,
        depth,
        proportion_rate,
        proportion_depth,
        fixed_cell=False,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        self["surface"] = surface
        self["rate"] = rate
        self["depth"] = depth
        if ("segment" in proportion_rate.dims) ^ ("segment" in proportion_depth.dims):
            raise ValueError(
                "Segment must be provided for both proportion_rate and"
                " proportion_depth, or for none at all."
            )
        self["proportion_rate"] = proportion_rate
        self["proportion_depth"] = proportion_depth
        self["fixed_cell"] = fixed_cell
        self["print_input"] = print_input
        self["print_flows"] = print_flows
        self["save_flows"] = save_flows
        self["observations"] = observations
        self._initialize_template()

        # TODO: add write logic for transforming proportion rate and depth to
        # the right shape in the binary file.
