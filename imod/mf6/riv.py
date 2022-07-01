import numpy as np

from imod.mf6.pkgbase import BoundaryCondition, VariableMetaData


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
    _metadata_dict = {
        "stage": VariableMetaData(np.floating),
        "conductance": VariableMetaData(np.floating, not_less_equal_than=0.0),
        "bottom_elevation": VariableMetaData(np.floating),
    }
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
        super().__init__(locals())
        self.dataset["stage"] = stage
        self.dataset["conductance"] = conductance
        self.dataset["bottom_elevation"] = bottom_elevation
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations

        self._pkgcheck_at_init()

    def _pkgcheck_at_init(self):
        self._check_bottom_above_stage()

        super()._pkgcheck_at_init()

    def _check_bottom_above_stage(self):
        """Check if river bottom not above river stage"""

        bottom_above_stage = self.dataset["bottom_elevation"] > self.dataset["stage"]

        if bottom_above_stage.any():
            raise ValueError(
                f"Bottom elevation above stage in {self.__class__.__name__}."
            )

    def _check_river_bottom_below_model_bottom(self, dis):
        """
        Check if river bottom not below model bottom. Modflow 6 throws an
        error if this occurs.
        """

        bottom = dis.dataset["bottom"]

        riv_below_bottom = self.dataset["bottom_elevation"] < bottom
        if riv_below_bottom.any():
            raise ValueError(
                f"River bottom below model bottom for in '{self.__class__.__name__}'."
            )

    def _pkgcheck_at_write(self, dis):
        self._check_river_bottom_below_model_bottom(self, dis)

        self._pkgcheck_at_write(dis)
