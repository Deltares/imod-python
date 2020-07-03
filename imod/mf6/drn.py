from imod.mf6.pkgbase import BoundaryCondition


class Drainage(BoundaryCondition):
    """
    The Drain package is used to simulate head-dependent flux boundaries.
    https://water.usgs.gov/ogw/modflow/mf6io.pdf#page=67

    Parameters
    ----------
    elevation: array of floats (xr.DataArray)
        elevation of the drain. (elev)
    conductance: array of floats (xr.DataArray)
        is the conductance of the drain. (cond)
    print_input: ({True, False}, optional)
        keyword to indicate that the list of drain information will be written
        to the listing file immediately after it is read. Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of drain flow rates will be printed to the
        listing file for every stress period time step in which “BUDGET PRINT”
        is specified in Output Control. If there is no Output Control option and
        PRINT FLOWS is specified, then flow rates are printed for the last time
        step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that drain flow terms will be written to the file specified
        with “BUDGET FILEOUT” in Output Control. Default is False.
    observations: [Not yet supported.]
        Default is None.
    """

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
    _period_data = ("elevation", "conductance")
    _keyword_map = {}
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
