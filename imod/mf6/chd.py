import numpy as np

from imod.mf6.pkgbase import BoundaryCondition
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA, CONC_DIMS_SCHEMA
from imod.schemata import (
    AllInsideNoDataSchema,
    AllNoDataSchema,
    AllValueSchema,
    CoordsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
    OtherCoordsSchema,
)


class ConstantHead(BoundaryCondition):
    """
    Constant-Head package. Any number of CHD Packages can be specified for a
    single groundwater flow model; however, an error will occur if a CHD Package
    attempts to make a GWF cell a constant-head cell when that cell has already
    been designated as a constant-head cell either within the present CHD
    Package or within another CHD Package. In previous MODFLOW versions, it was
    not possible to convert a constant-head cell to an active cell. Once a cell
    was designated as a constant-head cell, it remained a constant-head cell
    until the end of the end of the simulation. In MODFLOW 6 a constant-head
    cell will become active again if it is not included as a constant-head cell
    in subsequent stress periods. Previous MODFLOW versions allowed
    specification of SHEAD and EHEAD, which were the starting and ending
    prescribed heads for a stress period. Linear interpolation was used to
    calculate a value for each time step. In MODFLOW 6 only a single head value
    can be specified for any constant-head cell in any stress period. The
    time-series functionality must be used in order to interpolate values to
    individual time steps.

    Parameters
    ----------
    head: array of floats (xr.DataArray)
        Is the head at the boundary.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of constant head information will
        be written to the listing file immediately after it is read. Default is
        False.
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport, then this array is used
        as the  concentration for inflow over this boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport, then this keyword specifies
        how outflow over this boundary is computed.
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
    repeat_stress: Optional[xr.DataArray] of datetimes
        Used to repeat data for e.g. repeating stress periods such as
        seasonality without duplicating the values. The DataArray should have
        dimensions ``("repeat", "repeat_items")``. The ``repeat_items``
        dimension should have size 2: the first value is the "key", the second
        value is the "value". For the "key" datetime, the data of the "value"
        datetime will be used. Can also be set with a dictionary using the
        ``set_repeat_stress`` method.
    """

    _pkg_id = "chd"
    _keyword_map = {}
    _period_data = ("head",)

    _init_schemata = {
        "head": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
        ],
        "concentration": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            CONC_DIMS_SCHEMA,
        ],
    }
    _write_schemata = {
        "head": [
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "concentration": [IdentityNoDataSchema("head"), AllValueSchema(">=", 0.0)],
    }

    _keyword_map = {}
    _auxiliary_data = {"concentration": "species"}
    _template = BoundaryCondition._initialize_template(_pkg_id)

    _regrid_method = {
        "head": ("OverlapRegridder", "mean"),
        "concentration": ("OverlapRegridder", "mean"),
    }

    def __init__(
        self,
        head,
        concentration=None,
        concentration_boundary_type="aux",
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
        validate: bool = True,
        repeat_stress=None,
    ):
        super().__init__(locals())
        self.dataset["head"] = head
        if concentration is not None:
            self.dataset["concentration"] = concentration
            self.dataset["concentration_boundary_type"] = concentration_boundary_type
            self.add_periodic_auxiliary_variable()
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations
        self.dataset["repeat_stress"] = repeat_stress
        self._validate_init_schemata(validate)

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["head"] = self["head"]
        errors = super()._validate(schemata, **kwargs)

        return errors
