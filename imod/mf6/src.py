import numpy as np
from imod.mf6.pkgbase import DisStructuredBoundaryCondition, DisVerticesBoundaryCondition, VariableMetaData


class MassSourceLoadingDisStructured(DisStructuredBoundaryCondition):
    """
    SRC package for structured discretization (DIS) models .
    Any number of SRC Packages can be specified for a single groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.3.0.pdf#page=202

    Parameters
    ----------
    layer: list of int
        Model layer in which the well is located.
    row: list of int
        Row in which the well is located.
    column: list of int
        Column in which the well is located.
    smassrate: float or list of floats
        is the mass source loading rate. A positive value indicates addition of solute mass and a
        negative value indicates removal of solute mass.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of well information will be written to
        the listing file immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of well flow rates will be printed to the
        listing file for every stress period time step in which "BUDGET PRINT"
        is specified in Output Control. If there is no Output Control option
        and PRINT FLOWS is specified, then flow rates are printed for the last
        time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that well flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control.
        Default is False.
    observations: [Not yet supported.]
        Default is None.
    """

    _pkg_id = "src"
    _template = Package._initialize_template(_pkg_id)
    _period_data = ("layer", "row", "column", "rate")
    _keyword_map = {}

    _metadata_dict = {
        "layer": VariableMetaData(np.integer),
        "row": VariableMetaData(np.integer),
        "column": VariableMetaData(np.integer),
        "rate": VariableMetaData(np.floating),
    }

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
        super().__init__()
        self.dataset["layer"] = self.assign_dims(layer)
        self.dataset["row"] =  self.assign_dims(row)
        self.dataset["column"] =  self.assign_dims(column)
        self.dataset["rate"] =  self.assign_dims(rate)
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations
        self._pkgcheck()





class MassSourceLoadingDisVertices(DisVerticesBoundaryCondition):
    """
    SRC package for structured discretization (DIS) models .
    Any number of SRC Packages can be specified for a single groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.3.0.pdf#page=202

    Parameters
    ----------
    layer: list of int
        Model layer in which the well is located.
    row: list of int
        Row in which the well is located.
    column: list of int
        Column in which the well is located.
    smassrate: float or list of floats
        is the mass source loading rate. A positive value indicates addition of solute mass and a
        negative value indicates removal of solute mass.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of well information will be written to
        the listing file immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of well flow rates will be printed to the
        listing file for every stress period time step in which "BUDGET PRINT"
        is specified in Output Control. If there is no Output Control option
        and PRINT FLOWS is specified, then flow rates are printed for the last
        time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that well flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control.
        Default is False.
    observations: [Not yet supported.]
        Default is None.
    """

    _pkg_id = "src"
    _period_data = ("layer", "cell2d", "rate")
    _keyword_map = {}
    _template = DisVerticesBoundaryCondition._initialize_template(_pkg_id)

    def __init__(
        self,
        layer,
        cell2d,
        rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        super().__init__()
        self.dataset["layer"] =  self.assign_dims(layer)
        self.dataset["cell2d"] =  self.assign_dims(cell2d)
        self.dataset["rate"] =  self.assign_dims(rate)
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations


