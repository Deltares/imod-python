from dataclasses import asdict
from datetime import datetime
from typing import Optional

import numpy as np

from imod.logging import init_log_decorator
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.regrid.regrid_schemes import DrainageRegridMethod, RegridMethodType
from imod.mf6.utilities.regrid import (
    RegridderWeightsCache,
    _regrid_package_data,
)
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA, CONC_DIMS_SCHEMA
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION, allocate_drn_cells
from imod.prepare.topsystem.conductance import (
    DISTRIBUTING_OPTION,
    distribute_drn_conductance,
)
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
from imod.typing.grid import enforce_dim_order, is_planar_grid
from imod.util.expand_repetitions import expand_repetitions


class Drainage(BoundaryCondition, IRegridPackage):
    """
    The Drain package is used to simulate head-dependent flux boundaries.
    https://water.usgs.gov/ogw/modflow/mf6io.pdf#page=67

    Parameters
    ----------
    elevation: array of floats (xr.DataArray)
        elevation of the drain. (elev)
    conductance: array of floats (xr.DataArray)
        is the conductance of the drain. (cond)
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport, then this array is used
        as the  concentration for inflow over this boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport, then this keyword specifies
        how outflow over this boundary is computed.
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

    _pkg_id = "drn"

    # has to be ordered as in the list
    _init_schemata = {
        "elevation": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
        ],
        "conductance": [
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
        "elevation": [
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "conductance": [IdentityNoDataSchema("elevation"), AllValueSchema(">", 0.0)],
        "concentration": [IdentityNoDataSchema("elevation"), AllValueSchema(">=", 0.0)],
    }

    _period_data = ("elevation", "conductance")
    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}
    _regrid_method = DrainageRegridMethod()

    @init_log_decorator()
    def __init__(
        self,
        elevation,
        conductance,
        concentration=None,
        concentration_boundary_type="aux",
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
        validate: bool = True,
        repeat_stress=None,
    ):
        dict_dataset = {
            "elevation": elevation,
            "conductance": conductance,
            "concentration": concentration,
            "concentration_boundary_type": concentration_boundary_type,
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
        kwargs["elevation"] = self["elevation"]
        errors = super()._validate(schemata, **kwargs)

        return errors

    @classmethod
    def from_imod5_data(
        cls,
        key: str,
        imod5_data: dict[str, dict[str, GridDataArray]],
        period_data: dict[str, list[datetime]],
        target_discretization: StructuredDiscretization,
        target_npf: NodePropertyFlow,
        allocation_option: ALLOCATION_OPTION,
        distributing_option: DISTRIBUTING_OPTION,
        time_min: datetime,
        time_max: datetime,
        regridder_types: Optional[RegridMethodType] = None,
    ) -> "Drainage":
        """
        Construct a drainage-package from iMOD5 data, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        .. note::

            The method expects the iMOD5 model to be fully 3D, not quasi-3D.

        Parameters
        ----------
        imod5_data: dict
            Dictionary with iMOD5 data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        period_data: dict
            Dictionary with iMOD5 period data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        target_discretization:  StructuredDiscretization package
            The grid that should be used for the new package. Does not
            need to be identical to one of the input grids.
        target_npf: NodePropertyFlow package
            The conductivity information, used to compute drainage flux
        allocation_option: ALLOCATION_OPTION
            allocation option.
        distributing_option: dict[str, DISTRIBUTING_OPTION]
            distributing option.
        time_min: datetime
            Begin-time of the simulation. Used for expanding period data.
        time_max: datetime
            End-time of the simulation. Used for expanding period data.
        regridder_types: RegridMethodType, optional
            Optional dataclass with regridder types for a specific variable.
            Use this to override default regridding methods.

        Returns
        -------
        A Modflow 6 Drainage package.
        """

        target_top = target_discretization.dataset["top"]
        target_bottom = target_discretization.dataset["bottom"]
        target_idomain = target_discretization.dataset["idomain"]

        data = {
            "elevation": imod5_data[key]["elevation"],
            "conductance": imod5_data[key]["conductance"],
        }
        is_planar = is_planar_grid(data["elevation"])

        if regridder_types is None:
            regridder_settings = asdict(cls.get_regrid_methods(), dict_factory=dict)
        else:
            regridder_settings = asdict(regridder_types, dict_factory=dict)

        regrid_context = RegridderWeightsCache()

        regridded_package_data = _regrid_package_data(
            data, target_idomain, regridder_settings, regrid_context, {}
        )

        conductance = regridded_package_data["conductance"]

        if is_planar:
            planar_elevation = regridded_package_data["elevation"]

            drn_allocation = allocate_drn_cells(
                allocation_option,
                target_idomain == 1,
                target_top,
                target_bottom,
                planar_elevation,
            )

            layered_elevation = planar_elevation.where(drn_allocation)
            layered_elevation = enforce_dim_order(layered_elevation)
            regridded_package_data["elevation"] = layered_elevation

            regridded_package_data["conductance"] = distribute_drn_conductance(
                distributing_option,
                drn_allocation,
                conductance,
                target_top,
                target_bottom,
                target_npf.dataset["k"],
                planar_elevation,
            )

        drn = Drainage(**regridded_package_data)
        if period_data is not None:
            repeat = period_data.get(key)
            if repeat is not None:
                drn.set_repeat_stress(expand_repetitions(repeat, time_min, time_max))
        return drn
