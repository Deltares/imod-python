import numpy as np

from imod.mf6.pkgbase import BoundaryCondition, VariableMetaData


class GeneralHeadBoundary(BoundaryCondition):
    """
    The General-Head Boundary package is used to simulate head-dependent flux
    boundaries.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=75

    Parameters
    ----------
    head: array of floats (xr.DataArray)
        is the boundary head. (bhead)
    conductance: array of floats (xr.DataArray)
        is the hydraulic conductance of the interface between the aquifer cell and
        the boundary.(cond)
    print_input: ({True, False}, optional)
        keyword to indicate that the list of general head boundary information
        will be written to the listing file immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of general head boundary flow rates will be
        printed to the listing file for every stress period time step in which
        "BUDGET PRINT" is specified in Output Control. If there is no Output
        Control option and PRINT FLOWS is specified, then flow rates are printed
        for the last time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that general head boundary flow terms will be written to the
        file specified with "BUDGET FILEOUT" in Output Control.
        Default is False.
    observations: [Not yet supported.]
        Default is None.
    """

    _pkg_id = "ghb"
    _period_data = ("head", "conductance")
    _metadata_dict = {
        "head": VariableMetaData(np.floating),
        "conductance": VariableMetaData(np.floating),
    }
    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(
        self,
        head,
        conductance,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        super().__init__(locals())
        self.dataset["head"] = head
        self.dataset["conductance"] = conductance
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations

        self._pkgcheck()

    def _pkgcheck(self):
        self._check_conductance_zero()

        super()._pkgcheck()

    def _check_conductance_zero(self):
        """Check if conductance is zero, modflow 6 doesn't accept this."""
        
        conductance_zero = self.dataset["conductance"] == 0.0
        if conductance_zero.any():
            raise ValueError(
                f"Detected conductance with value 0. in {self.__class__.__name__}"
            )
