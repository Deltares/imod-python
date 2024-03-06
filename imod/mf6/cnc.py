import numpy as np

from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA
from imod.schemata import (
    AllInsideNoDataSchema,
    AllNoDataSchema,
    AllValueSchema,
    CoordsSchema,
    DTypeSchema,
    IndexesSchema,
    OtherCoordsSchema,
)
from imod.mf6.interfaces.iregridpackage import IRegridPackage

class ConstantConcentration(BoundaryCondition, IRegridPackage):
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
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
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
            BOUNDARY_DIMS_SCHEMA,
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
        validate: bool = True,
    ):
        dict_dataset = {
            "concentration": concentration,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "observations": observations,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)
