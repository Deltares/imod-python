from typing import Dict

import numpy as np
import xarray as xr

from imod.mf6.pkgbase import BoundaryCondition


def assign_dims(arg) -> Dict:
    is_da = isinstance(arg, xr.DataArray)
    if is_da and "time" in arg.coords:
        if arg.ndim != 2:
            raise ValueError("time varying variable: must be 2d")
        if arg.dims[0] != "time":
            arg = arg.transpose()
        da = xr.DataArray(
            data=arg.values, coords={"time": arg["time"]}, dims=["time", "index"]
        )
        return da
    elif is_da:
        return ("index", arg.values)
    else:
        return ("index", arg)


class WellDisStructured(BoundaryCondition):
    """
    WEL package for structured discretization (DIS) models .
    Any number of WEL Packages can be specified for a single groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=63

    Parameters
    ----------
    layer: list of int
        Model layer in which the well is located.
    row: list of int
        Row in which the well is located.
    column: list of int
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

    _pkg_id = "wel"
    _period_data = ("layer", "row", "column", "rate")
    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)

    _expected_dtypes = {
        "layer": np.integer,
        "row": np.integer,
        "column": np.integer,
        "rate": np.floating,
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
        self.dataset["layer"] = assign_dims(layer)
        self.dataset["row"] = assign_dims(row)
        self.dataset["column"] = assign_dims(column)
        self.dataset["rate"] = assign_dims(rate)
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations

        self._pkgcheck()

    def to_sparse(self, arrdict, layer):
        spec = []
        for key in arrdict:
            if key in ["layer", "row", "column"]:
                spec.append((key, np.int32))
            else:
                spec.append((key, np.float64))

        sparse_dtype = np.dtype(spec)
        nrow = next(iter(arrdict.values())).size
        recarr = np.empty(nrow, dtype=sparse_dtype)
        for key, arr in arrdict.items():
            recarr[key] = arr
        return recarr


class WellDisVertices(BoundaryCondition):
    """
    WEL package for discretization by vertices (DISV) models.
    Any number of WEL Packages can be specified for a single groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=63

    Parameters
    ----------
    layer: list of int
        Modellayer in which the well is located.
    cell2d: list of int
        Cell in which the well is located.
    rate: float or list of floats
        is the volumetric well rate. A positive value indicates well
        (injection) and a negative value indicates discharge (extraction) (q).
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

    _pkg_id = "wel"
    _period_data = ("layer", "cell2d", "rate")
    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)

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
        self.dataset["layer"] = assign_dims(layer)
        self.dataset["cell2d"] = assign_dims(cell2d)
        self.dataset["rate"] = assign_dims(rate)
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations

    def to_sparse(self, arrdict, layer):
        spec = []
        for key in arrdict:
            if key in ["layer", "cell2d"]:
                spec.append((key, np.int32))
            else:
                spec.append((key, np.float64))

        sparse_dtype = np.dtype(spec)
        nrow = next(iter(arrdict.values())).size
        recarr = np.empty(nrow, dtype=sparse_dtype)
        for key, arr in arrdict.items():
            recarr[key] = arr
        return recarr
