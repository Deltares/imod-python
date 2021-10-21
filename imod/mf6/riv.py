from imod.mf6.pkgbase import BoundaryCondition


class River(BoundaryCondition):
    """
    River package.
    Any number of RIV Packages can be specified for a single groundwater flow
    model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=71

    Parameters
    ----------
    stage: array of floats (xr.DataArray)
        is the head in the river.
    conductance: array of floats (xr.DataArray)
        is the riverbed hydraulic conductance.
    bottom_elevation: array of floats (xr.DataArray)
        is the elevation of the bottom of the riverbed.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of drain information will be written
        to the listing file immediately after it is read. Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of drain flow rates will be printed to the
        listing file for every stress period time step in which "BUDGET PRINT"
        is specified in Output Control. If there is no Output Control option and
        PRINT FLOWS is specified, then flow rates are printed for the last time
        step of each stress period. Default is False.
    save_flows: ({True, False}, optional)
        Indicates that drain flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control. Default is False.
    observations: [Not yet supported.]
        Default is None.
    """

    _pkg_id = "riv"
    _period_data = ("stage", "conductance", "bottom_elevation")
    _keyword_map = {}
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
        self.dataset["stage"] = stage
        self.dataset["conductance"] = conductance
        self.dataset["bottom_elevation"] = bottom_elevation
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations
