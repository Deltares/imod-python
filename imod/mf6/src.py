import numpy as np

from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.pkgbase import Package
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA
from imod.schemata import (
    AllInsideNoDataSchema,
    AllNoDataSchema,
    DTypeSchema,
    IndexesSchema,
    OtherCoordsSchema,
)


class MassSourceLoading(BoundaryCondition):
    """
    Mass Source Loading (SRC) package for structured discretization (DIS)
    models. Any number of SRC Packages can be specified for a single
    groundwater flow model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.3.0.pdf#page=202

    Parameters
    ----------
    rate: xr.DataArray of floats
        is the mass source loading rate. A positive value indicates addition of
        solute mass and a negative value indicates removal of solute mass
        (smassrate).
    print_input: ({True, False}, optional), default is False.
        keyword to indicate that the list of mass source information will be
        written to the listing file immediately after it is read.
    print_flows: ({True, False}, optional), default is False.
        keyword to indicate that the list of mass source flow rates will be
        printed to the listing file for every stress period time step in which
        "BUDGET PRINT" is specified in Output Control. If there is no Output
        Control option and "PRINT FLOWS" is specified, then flow rates are
        printed for the last time step of each stress period.
    save_flows: ({True, False}, optional)
        Indicates that the mass source flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control.
        Default is False.
    observations: [Not yet supported.]
        Default is None.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "src"
    _template = Package._initialize_template(_pkg_id)
    _period_data = ("rate",)
    _keyword_map = {}

    _init_schemata = {
        "rate": (
            DTypeSchema(np.floating),
            IndexesSchema(),
            BOUNDARY_DIMS_SCHEMA,
        )
    }

    _write_schemata = {
        "rate": [
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
    }

    def __init__(
        self,
        rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
        validate: bool = True,
    ):
        super().__init__()
        self.dataset["rate"] = rate
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations
        self._validate_init_schemata(validate)
