import numpy as np

from imod.mf6.pkgbase import BoundaryCondition
from imod.mf6.validation import BC_DIMS_SCHEMA
from imod.schemata import (
    AllInsideNoDataSchema,
    AllNoDataSchema,
    AllValueSchema,
    CoordsSchema,
    DTypeSchema,
    IndexesSchema,
    OtherCoordsSchema,
)


class ConstantConcentration(BoundaryCondition):
    """
    Constant Concentration package.

    Parameters
    ----------
    concentration: array of floats (xr.DataArray)
        Concentration of the boundary.
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
    _template = BoundaryCondition._initialize_template(_pkg_id)

    _init_schemata = {
        "concentration": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BC_DIMS_SCHEMA,
        ],
    }
    _write_schemata = {
        "concentration": [
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
            AllValueSchema(">=", 0.0),
        ]
    }

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
