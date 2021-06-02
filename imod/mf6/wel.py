import numpy as np

from imod.mf6.pkgbase import BoundaryCondition


class Well(BoundaryCondition):
    """
    WEL package.
    Any number of WEL Packages can be specified for a single groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=63

    Parameters
    ----------
    layer: int or list of int
        Modellayer in which the well is located.
    row: int or list of int
        Row in which the well is located.
    column: int or list of int
        Column in which the well is located.
    rate: float or list of floats
        is the volumetric well rate. A positive value indicates well
        (injection) and a negative value indicates discharge (extraction) (q).
    print_input: ({True, False}, optional)
        keyword to indicate that the list of well information will be written to
        the listing file immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of well flow rates will be printed to the
        listing file for every stress period time step in which "BUDGET PRINT"is
        specified in Output Control. If there is no Output Control option and
        PRINT FLOWS is specified, then flow rates are printed for the last time
        step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that well flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control.
        Default is False.
    observations: [Not yet supported.]
        Default is None.
    """

    __slots__ = (
        "layer",
        "row",
        "column",
        "rate",
        "print_input",
        "print_flows",
        "save_flows",
        "observations",
    )
    _pkg_id = "wel"
    _period_data = ("layer", "row", "column", "rate")
    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)

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
        index = np.arange(len(layer))
        self["index"] = index
        self["layer"] = ("index", layer)
        self["row"] = ("index", row)
        self["column"] = ("index", column)
        self["rate"] = ("index", rate)
        self["print_input"] = print_input
        self["print_flows"] = print_flows
        self["save_flows"] = save_flows
        self["observations"] = observations

    def to_sparse(self, arrlist, *args, **kwargs):
        nrow = arrlist[0].size
        listarr = np.empty((nrow, 5), np.int32)
        listarr[:, 0] = arrlist[0]
        listarr[:, 1] = arrlist[1]
        listarr[:, 2] = arrlist[2]
        values = arrlist[3].astype(np.float64)
        listarr[:, 3:5] = values.reshape(nrow, 1).view(np.int32)
        # flatten to 1D such that numpy tofile doesn't write extra array dims
        return listarr.flatten()
