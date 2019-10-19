import numpy as np

from imod.mf6.pkgbase import BoundaryCondition


class Well(BoundaryCondition):
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
    _binary_data = ("layer", "row", "column", "rate")
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
