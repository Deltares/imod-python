from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from imod.logging import init_log_decorator
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.disv import VerticesDiscretization
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.utilities.regrid import (
    RegridderType,
    RegridderWeightsCache,
    _regrid_package_data,
)
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA, CONC_DIMS_SCHEMA
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION, allocate_rch_cells
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
from imod.typing import GridDataArray
from imod.typing.grid import (
    enforce_dim_order,
    is_planar_grid,
)


class Recharge(BoundaryCondition, IRegridPackage):
    """
    Recharge Package.
    Any number of RCH Packages can be specified for a single groundwater flow
    model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=79

    Parameters
    ----------
    rate: array of floats (xr.DataArray)
        is the recharge flux rate (LT −1). This rate is multiplied inside the
        program by the surface area of the cell to calculate the volumetric
        recharge rate. A time-series name may be specified.
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport, then this array is used
        as the  concentration for inflow over this boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport, then this keyword specifies
        how outflow over this boundary is computed.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of recharge information will be
        written to the listing file immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of recharge flow rates will be printed to the
        listing file for every stress period time step in which "BUDGET PRINT"is
        specified in Output Control. If there is no Output Control option and
        PRINT FLOWS is specified, then flow rates are printed for the last time
        step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that recharge flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control.
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
    fixed_cell: ({True, False}, optional)
        indicates that recharge will not be reassigned to a cell underlying the
        cell specified in the list if the specified cell is inactive.
    """

    _pkg_id = "rch"
    _period_data = ("rate",)
    _keyword_map = {}

    _init_schemata = {
        "rate": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
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
        "rate": [
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "concentration": [IdentityNoDataSchema("rate"), AllValueSchema(">=", 0.0)],
    }

    _template = BoundaryCondition._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}

    _regrid_method = {
        "rate": (RegridderType.OVERLAP, "mean"),
        "concentration": (RegridderType.OVERLAP, "mean"),
    }

    @init_log_decorator()
    def __init__(
        self,
        rate,
        concentration=None,
        concentration_boundary_type="auxmixed",
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
        validate: bool = True,
        repeat_stress=None,
        fixed_cell: bool = False,
    ):
        dict_dataset = {
            "rate": rate,
            "concentration": concentration,
            "concentration_boundary_type": concentration_boundary_type,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "observations": observations,
            "repeat_stress": repeat_stress,
            "fixed_cell": fixed_cell,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["rate"] = self["rate"]
        errors = super()._validate(schemata, **kwargs)

        return errors

    def get_regrid_methods(self) -> Optional[dict[str, Tuple[RegridderType, str]]]:
        return self._regrid_method

    @classmethod
    def from_imod5_data(
        cls,
        imod5_data: dict[str, dict[str, GridDataArray]],
        discretization_package: VerticesDiscretization | StructuredDiscretization,
        regridder_types: Optional[dict[str, tuple[RegridderType, str]]] = None,
    ) -> "Recharge":
        """
        Construct an rch-package from iMOD5 data, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        .. note::

            The method expects the iMOD5 model to be fully 3D, not quasi-3D.

        Parameters
        ----------
        imod5_data: dict
            Dictionary with iMOD5 data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        target_grid: GridDataArray
            The grid that should be used for the new package. Does not
            need to be identical to one of the input grids.
        regridder_types: dict, optional
            Optional dictionary with regridder types for a specific variable.
            Use this to override default regridding methods.

        Returns
        -------
        Modflow 6 rch package.

        """
        new_idomain = discretization_package.dataset["idomain"]
        data = {
            "rate": imod5_data["rch"]["rate"],
        }
        new_package_data = {}

        # first regrid the inputs to the target grid.
        regridder_settings = deepcopy(cls._regrid_method)
        if regridder_types is not None:
            regridder_settings.update(regridder_types)

        regrid_context = RegridderWeightsCache()

        new_package_data = _regrid_package_data(
            data, new_idomain, regridder_settings, regrid_context, {}
        )

        # if rate has only layer 0, then it is planar.
        if is_planar_grid(new_package_data["rate"]):
            planar_rate_regridded = new_package_data["rate"].isel(layer=0, drop=True)
            # create an array indicating in which cells rch is active
            is_rch_cell = allocate_rch_cells(
                ALLOCATION_OPTION.at_first_active,
                new_idomain,
                planar_rate_regridded,
            )

            # remove rch from cells where it is not allocated and broadcast over layers.
            rch_rate = planar_rate_regridded.where(is_rch_cell)
            rch_rate = enforce_dim_order(rch_rate)
            new_package_data["rate"] = rch_rate

        return Recharge(**new_package_data)
