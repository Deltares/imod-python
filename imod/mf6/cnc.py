import numpy as np

from imod.mf6.pkgbase import BoundaryCondition, VariableMetaData


class ConstantConcentration(BoundaryCondition):
    """
    Constant-Concentration package.
    Parameters
    ----------
    concentration: array of floats (xr.DataArray)
        Is the concentration at the boundary.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of constant head information will
        be written to the listing file immediately after it is read. Default is
        False.
    print_flows: ({True, False}, optional)
        Indicates that the list of constant head flow rates will be printed to
        the listing file for every stress period time step in which "BUDGET
        PRINT" is specified in Output Control. If there is no Output Control
        option and PRINT FLOWS is specified, then flow rates are printed for the
        last time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that constant head flow terms will be written to the file
        specified with "BUDGET FILEOUT" in Output Control. Default is False.
    observations: [Not yet supported.]
        Default is None.
    """

    _pkg_id = "cnc"
    _keyword_map = {}
    _period_data = ("concentration",)
    _metadata_dict = {"concentration": VariableMetaData(np.floating)}
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(
        self,
        concentration,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        super().__init__(locals())
        self.dataset["concentration"] = concentration
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations

        self._pkgcheck()
