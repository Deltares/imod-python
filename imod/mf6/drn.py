import numpy as np

from imod.mf6.pkgbase import BoundaryCondition, VariableMetaData
from imod.mf6.validation import BC_DIMS_SCHEMA
from imod.schemata import (
    AllInsideNoDataSchema,
    AllNoDataSchema,
    AllValueSchema,
    CoordsSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
    OtherCoordsSchema,
)


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
        listing file for every stress period time step in which "BUDGET PRINT"
        is specified in Output Control. If there is no Output Control option and
        PRINT FLOWS is specified, then flow rates are printed for the last time
        step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that drain flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control. Default is False.
    observations: [Not yet supported.]
        Default is None.
    """

    _pkg_id = "drn"

    # has to be ordered as in the list
    _init_schemata = {
        "elevation": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BC_DIMS_SCHEMA,
        ],
        "conductance": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BC_DIMS_SCHEMA,
        ],
        "print_flows": [DTypeSchema(np.bool_), DimsSchema()],
        "save_flows": [DTypeSchema(np.bool_), DimsSchema()],
    }
    _write_schemata = {
        "elevation": [
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "conductance": [IdentityNoDataSchema("elevation"), AllValueSchema(">", 0.0)],
    }

    _metadata_dict = {
        "elevation": VariableMetaData(np.floating),
        "conductance": VariableMetaData(np.floating, not_less_equal_than=0.0),
    }
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
        super().__init__(locals())
        self.dataset["elevation"] = elevation
        self.dataset["conductance"] = conductance
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations

        self._validate_at_init()

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["elevation"] = self["elevation"]
        errors = super()._validate(schemata, **kwargs)

        return errors
