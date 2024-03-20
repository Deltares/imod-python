from typing import Optional, Tuple

import numpy as np

from imod.logging.logging_decorators import init_log_decorator
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.utilities.regrid import RegridderType
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA, CONC_DIMS_SCHEMA
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
from imod.util.spatial import unstack_dim_into_variable

SEGMENT_BOUNDARY_DIMS_SCHEMA = (
    BOUNDARY_DIMS_SCHEMA
    | DimsSchema("segment", "time", "layer", "y", "x")
    | DimsSchema("segment", "layer", "y", "x")
    | DimsSchema("segment", "time", "layer", "{face_dim}")
    | DimsSchema("segment", "layer", "{face_dim}")
    # Layer dim not necessary, as long as there is a layer coordinate present.
    | DimsSchema("segment", "time", "y", "x")
    | DimsSchema("segment", "y", "x")
    | DimsSchema("segment", "time", "{face_dim}")
    | DimsSchema("segment", "{face_dim}")
)

from imod.mf6.interfaces.iregridpackage import IRegridPackage


class Evapotranspiration(BoundaryCondition, IRegridPackage):
    """
    Evapotranspiration (EVT) Package.
    Any number of EVT Packages can be specified for a single groundwater flow
    model. All single-valued variables are free format.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=86

    Parameters
    ----------
    surface: array of floats (xr.DataArray)
        is the elevation of the ET surface (L). A time-series name may be
        specified.
    rate: array of floats (xr.DataArray)
        is the maximum ET flux rate (LT −1). A time-series name may be
        specified.
    depth: array of floats (xr.DataArray)
        is the ET extinction depth (L). A time-series name may be specified.
    proportion_rate: array of floats (xr.DataArray)
        is the proportion of the maximum ET flux rate at the bottom of a segment
        (dimensionless). A time-series name may be specified. (petm)
    proportion_depth: array of floats (xr.DataArray)
        is the proportion of the ET extinction depth at the bottom of a segment
        (dimensionless). A timeseries name may be specified. (pxdp)
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport, then this array is used
        as the  concentration for inflow over this boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport, then this keyword specifies
        how outflow over this boundary is computed.
    fixed_cell: array of floats (xr.DataArray)
        indicates that evapotranspiration will not be reassigned to a cell
        underlying the cell specified in the list if the specified cell is
        inactive.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of evapotranspiration information will
        be written to the listing file immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of evapotranspiration flow rates will be printed
        to the listing file for every stress period time step in which "BUDGET
        PRINT" is specified in Output Control. If there is no Output Control
        option and PRINT FLOWS is specified, then flow rates are printed for the
        last time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that evapotranspiration flow terms will be written to the file
        specified with "BUDGET FILEOUT" in Output Control.
        Default is False.
    observations: [Not yet supported.]
        Default is None.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    repeat_stress: Optional[xr.DataArray] of datetimes
        Used to repeat data for e.g. repeating stress periods such as
        seasonality without duplicating the values. The DataArray should have
        dimensions ``("repeat", "repeat_items")``. The ``repeat_items``
        dimension should have size 2: the first value is the "key", the second
        value is the "value". For the "key" datetime, the data of the "value"
        datetime will be used. Can also be set with a dictionary using the
        ``set_repeat_stress`` method.
    """

    _pkg_id = "evt"
    _init_schemata = {
        "surface": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
        ],
        "rate": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
        ],
        "depth": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
        ],
        "proportion_rate": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            SEGMENT_BOUNDARY_DIMS_SCHEMA,
        ],
        "proportion_depth": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            SEGMENT_BOUNDARY_DIMS_SCHEMA,
        ],
        "concentration": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(
                (
                    "species",
                    "layer",
                )
            ),
            CONC_DIMS_SCHEMA,
        ],
        "print_flows": [DTypeSchema(np.bool_), DimsSchema()],
        "save_flows": [DTypeSchema(np.bool_), DimsSchema()],
    }
    _write_schemata = {
        "surface": [
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "rate": [IdentityNoDataSchema("surface")],
        "depth": [IdentityNoDataSchema("surface")],
        "proportion_rate": [IdentityNoDataSchema("surface")],
        "proportion_depth": [
            IdentityNoDataSchema("surface"),
            AllValueSchema(">=", 0.0),
            AllValueSchema("<=", 1.0),
        ],
        "concentration": [IdentityNoDataSchema("surface"), AllValueSchema(">=", 0.0)],
    }

    _period_data = ("surface", "rate", "depth", "proportion_depth", "proportion_rate")
    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}

    _regrid_method = {
        "surface": (
            RegridderType.OVERLAP,
            "mean",
        ),
        "rate": (
            RegridderType.OVERLAP,
            "mean",
        ),
        "depth": (
            RegridderType.OVERLAP,
            "mean",
        ),
        "proportion_rate": (
            RegridderType.OVERLAP,
            "mean",
        ),
        "proportion_depth": (
            RegridderType.OVERLAP,
            "mean",
        ),
    }
    @init_log_decorator()
    def __init__(
        self,
        surface,
        rate,
        depth,
        proportion_rate,
        proportion_depth,
        concentration=None,
        concentration_boundary_type="auxmixed",
        fixed_cell=False,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
        validate: bool = True,
        repeat_stress=None,
    ):
        if ("segment" in proportion_rate.dims) ^ ("segment" in proportion_depth.dims):
            raise ValueError(
                "Segment must be provided for both proportion_rate and"
                " proportion_depth, or for none at all."
            )
        dict_dataset = {
            "surface": surface,
            "rate": rate,
            "depth": depth,
            "proportion_rate": proportion_rate,
            "proportion_depth": proportion_depth,
            "concentration": concentration,
            "concentration_boundary_type": concentration_boundary_type,
            "fixed_cell": fixed_cell,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "observations": observations,
            "repeat_stress": repeat_stress,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["surface"] = self["surface"]
        errors = super()._validate(schemata, **kwargs)

        return errors

    def _get_options(
        self, predefined_options: dict, not_options: Optional[list] = None
    ):
        options = super()._get_options(predefined_options, not_options=not_options)
        # Add amount of segments
        if "segment" in self.dataset.dims:
            options["nseg"] = self.dataset.dims["segment"] + 1
        else:
            options["nseg"] = 1
        return options

    def _get_bin_ds(self):
        bin_ds = super()._get_bin_ds()

        # Unstack "segment" dimension into different variables
        bin_ds = unstack_dim_into_variable(bin_ds, "segment")

        return bin_ds
    
    def get_regrid_methods(self) -> Optional[dict[str, Tuple[RegridderType, str]]]:
        return self._regrid_method