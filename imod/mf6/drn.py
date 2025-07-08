from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np

from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.common.utilities.dataclass_type import DataclassType
from imod.common.utilities.mask import broadcast_and_mask_arrays
from imod.logging import init_log_decorator, standard_log_decorator
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.disv import VerticesDiscretization
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.regrid.regrid_schemes import DrainageRegridMethod
from imod.mf6.aggregate.aggregate_schemes import DrainageAggregationMethod
from imod.mf6.utilities.imod5_converter import regrid_imod5_pkg_data
from imod.mf6.utilities.package import set_repeat_stress_if_available
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA, CONC_DIMS_SCHEMA
from imod.prepare.cleanup import cleanup_drn
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION, allocate_drn_cells
from imod.prepare.topsystem.conductance import (
    DISTRIBUTING_OPTION,
    distribute_drn_conductance,
)
from imod.schemata import (
    AllCoordsValueSchema,
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
from imod.typing.grid import enforce_dim_order, has_negative_layer, is_planar_grid
from imod.util.regrid import RegridderWeightsCache


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
    repeat_stress: dict or xr.DataArray of datetimes, optional
        Used to repeat data for e.g. repeating stress periods such as
        seasonality without duplicating the values. If provided as dict, it
        should map new dates to old dates present in the dataset.
        ``{"2001-04-01": "2000-04-01", "2001-10-01": "2000-10-01"}`` if provided
        as DataArray, it should have dimensions ``("repeat", "repeat_items")``.
        The ``repeat_items`` dimension should have size 2: the first value is
        the "key", the second value is the "value". For the "key" datetime, the
        data of the "value" datetime will be used.
    """

    _pkg_id = "drn"

    # has to be ordered as in the list
    _init_schemata = {
        "elevation": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "conductance": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
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
            AllCoordsValueSchema("layer", ">", 0),
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
    _aggregate_method: DataclassType = DrainageAggregationMethod()

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

    @standard_log_decorator()
    def cleanup(self, dis: StructuredDiscretization | VerticesDiscretization) -> None:
        """
        Clean up package inplace. This method calls
        :func:`imod.prepare.cleanup_drn`, see documentation of that
        function for details on cleanup.

        dis: imod.mf6.StructuredDiscretization | imod.mf6.VerticesDiscretization
            Model discretization package.
        """
        dis_dict = {"idomain": dis.dataset["idomain"]}
        cleaned_dict = self._call_func_on_grids(cleanup_drn, dis_dict)
        super().__init__(cleaned_dict)

    @classmethod
    def allocate_and_distribute_planar_data(
        cls,
        planar_data: dict[str, GridDataArray],
        dis: StructuredDiscretization,
        npf: NodePropertyFlow,
        allocation_option: ALLOCATION_OPTION,
        distributing_option: DISTRIBUTING_OPTION,
    ) -> dict[str, GridDataArray]:
        """
        Allocate and distribute planar data for given discretization and npf
        package. If layer number of ``planar_data`` is negative,
        ``allocation_option`` is overrided and set to
        ALLOCATION_OPTION.at_first_active.

        Parameters
        ----------
        planar_data: dict[str, GridDataArray]
            Dictionary with planar grid data.
        dis: imod.mf6.StructuredDiscretization
            Model discretization package.
        npf: imod.mf6.NodePropertyFlow
            Node property flow package.
        allocation_option: ALLOCATION_OPTION
            allocation option. If planar data is assigned to a negative layer
            number, this option is overridden and set to
            ALLOCATION_OPTION.at_first_active.
        distributing_option: DISTRIBUTING_OPTION
            distributing option.

        Returns
        -------
        dict[str, GridDataArray]
            Dictionary with layered grid data.
        """

        top = dis.dataset["top"]
        bottom = dis.dataset["bottom"]
        idomain = dis.dataset["idomain"]

        if has_negative_layer(planar_data["elevation"]):
            allocation_option = ALLOCATION_OPTION.at_first_active

        # Enforce planar data, remove all layer dimension information
        planar_data = {
            key: grid.isel({"layer": 0}, drop=True, missing_dims="ignore")
            for key, grid in planar_data.items()
        }

        drn_allocation = allocate_drn_cells(
            allocation_option,
            idomain > 0,
            top,
            bottom,
            planar_data["elevation"],
        )
        layered_data = {}
        layered_data["conductance"] = distribute_drn_conductance(
            distributing_option,
            drn_allocation,
            planar_data["conductance"],
            top,
            bottom,
            npf.dataset["k"],
            planar_data["elevation"],
        )
        layered_data["elevation"] = planar_data["elevation"].where(drn_allocation)
        layered_data["elevation"] = enforce_dim_order(layered_data["elevation"])
        return layered_data

    @classmethod
    def from_imod5_data(
        cls,
        key: str,
        imod5_data: dict[str, dict[str, GridDataArray]],
        period_data: dict[str, list[datetime]],
        target_dis: StructuredDiscretization,
        target_npf: NodePropertyFlow,
        time_min: datetime,
        time_max: datetime,
        allocation_option: ALLOCATION_OPTION,
        distributing_option: DISTRIBUTING_OPTION,
        regridder_types: Optional[DrainageRegridMethod] = None,
        regrid_cache: RegridderWeightsCache = RegridderWeightsCache(),
    ) -> "Drainage":
        """
        Construct a drainage-package from iMOD5 data, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        .. note::

            The method expects the iMOD5 model to be fully 3D, not quasi-3D.

        Parameters
        ----------
        key: str
            Packagename of the iMOD5 data to use.
        imod5_data: dict
            Dictionary with iMOD5 data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        period_data: dict
            Dictionary with iMOD5 period data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        target_dis:  StructuredDiscretization package
            The grid that should be used for the new package. Does not
            need to be identical to one of the input grids.
        target_npf: NodePropertyFlow package
            The conductivity information, used to compute drainage flux
        allocation_option: ALLOCATION_OPTION
            allocation option. If package data is assigned to a negative layer
            number, this option is overridden and set to
            ALLOCATION_OPTION.at_first_active.
        distributing_option: dict[str, DISTRIBUTING_OPTION]
            distributing option.
        time_min: datetime
            Begin-time of the simulation. Used for expanding period data.
        time_max: datetime
            End-time of the simulation. Used for expanding period data.
        regridder_types: DrainageRegridMethod, optional
            Optional dataclass with regridder types for a specific variable.
            Use this to override default regridding methods.
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to speed up regridding,
            if the same regridders are used several times for regridding different arrays.

        Returns
        -------
        A Modflow 6 Drainage package.
        """
        data = {
            "elevation": imod5_data[key]["elevation"],
            "conductance": imod5_data[key]["conductance"],
        }
        mask = data["conductance"] > 0
        data["conductance"] = data["conductance"].where(mask)
        # Regrid the input data
        regridded_package_data = regrid_imod5_pkg_data(
            cls, data, target_dis, regridder_types, regrid_cache
        )
        regridded_package_data = broadcast_and_mask_arrays(regridded_package_data)
        is_planar = is_planar_grid(regridded_package_data["elevation"])
        if is_planar:
            layered_data = cls.allocate_and_distribute_planar_data(
                regridded_package_data,
                target_dis,
                target_npf,
                allocation_option,
                distributing_option,
            )
            regridded_package_data.update(layered_data)

        drn = cls(**regridded_package_data, validate=True)
        repeat = period_data.get(key)
        set_repeat_stress_if_available(repeat, time_min, time_max, drn)
        # Clip the drain package to the time range of the simulation and ensure
        # time is forward filled.
        drn = drn.clip_box(time_min=time_min, time_max=time_max)

        return drn

    @classmethod
    def get_regrid_methods(cls) -> DrainageRegridMethod:
        return deepcopy(cls._regrid_method)
