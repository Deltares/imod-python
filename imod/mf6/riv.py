from datetime import datetime
from typing import Optional, Tuple, cast

import numpy as np

from imod import logging
from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.common.utilities.dataclass_type import DataclassType
from imod.common.utilities.mask import broadcast_and_mask_arrays
from imod.logging import init_log_decorator, standard_log_decorator
from imod.mf6.aggregate.aggregate_schemes import RiverAggregationMethod
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.disv import VerticesDiscretization
from imod.mf6.drn import Drainage
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.regrid.regrid_schemes import RiverRegridMethod
from imod.mf6.topsystem import TopSystemBoundaryCondition
from imod.mf6.utilities.imod5_converter import regrid_imod5_pkg_data
from imod.mf6.utilities.package import set_repeat_stress_if_available
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA, CONC_DIMS_SCHEMA
from imod.prepare.cleanup import AlignLevelsMode, align_interface_levels, cleanup_riv
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION, allocate_riv_cells
from imod.prepare.topsystem.conductance import (
    DISTRIBUTING_OPTION,
    distribute_drn_conductance,
    distribute_riv_conductance,
    split_conductance_with_infiltration_factor,
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
from imod.typing import GridDataArray, GridDataDict
from imod.typing.grid import (
    concat,
    enforce_dim_order,
    has_negative_layer,
    is_planar_grid,
)
from imod.util.regrid import (
    RegridderWeightsCache,
)


def mask_package__drop_if_empty(
    package: TopSystemBoundaryCondition,
) -> Optional[TopSystemBoundaryCondition]:
    """ "
    Create an optional package from a package if it has data. Return None if
    package is inactive everywhere.
    """
    # remove River package if its mask is False everywhere
    mask = ~np.isnan(package["conductance"])
    return package.mask(mask) if np.any(mask) else None


def clip_time_if_package(
    package: Optional[TopSystemBoundaryCondition],
    time_min: datetime,
    time_max: datetime,
) -> Optional[TopSystemBoundaryCondition]:
    if package is not None:
        package = package.clip_box(time_min=time_min, time_max=time_max)
    return package


def rise_bottom_elevation_if_needed(
    bottom_elevation: GridDataArray, bottom: GridDataArray
) -> GridDataArray:
    """
    Due to regridding, the bottom_elevation could be less than the
    layer bottom, so here we overwrite it with bottom if that's
    the case.
    """
    is_layer_bottom_above_bottom_elevation = (bottom > bottom_elevation).any()

    if is_layer_bottom_above_bottom_elevation:
        logging.logger.warning(
            "Note: riv bottom was detected below model bottom. Updated the riv's bottom."
        )
        bottom_elevation, _ = align_interface_levels(
            bottom_elevation, bottom, AlignLevelsMode.BOTTOMUP
        )
    return bottom_elevation


def _separate_infiltration_data(
    riv_pkg_data: GridDataDict, infiltration_factor: GridDataArray
) -> tuple[GridDataDict, GridDataDict]:
    """
    Account for the infiltration factor in the river package data. This function
    updates the riv_pkg_data with an infiltration conductance. The extra
    exfiltration conductance is separated into a data dict for drainage
    """
    # update the conductance of the river package to account for the
    # infiltration factor
    drain_conductance, river_conductance = split_conductance_with_infiltration_factor(
        riv_pkg_data["conductance"], infiltration_factor
    )
    riv_pkg_data["conductance"] = river_conductance
    # create a drainage package with the conductance we computed from the
    # infiltration factor
    drn_pkg_data = {
        "elevation": riv_pkg_data["stage"],
        "conductance": drain_conductance,
    }
    return riv_pkg_data, drn_pkg_data


def _create_drain_from_leftover_riv_imod5_data(
    allocation_drn_data: GridDataDict,
    infiltration_drn_data: GridDataDict,
) -> Drainage:
    """
    Create a drainage package from leftover imod5 river package data,
    stemming from:

        * If ``ALLOCATION_OPTION.stage_to_riv_bottom_drn_above`` is chosen,
            drain cells are allocated from the first active cell to river
            stage. In this case ``allocation_drn_data`` is not empty.
        * Infiltration factor. This factor is optional in imod5, but it
            does not exist in MF6, so we mimic its effect with a Drainage
            boundary. This data is stored in ``infiltration_drn_data``.
    """

    if allocation_drn_data:
        drain_leftover_data: GridDataDict = {}
        for key, allocation_grid in allocation_drn_data.items():
            concatenated = concat(
                [allocation_grid, infiltration_drn_data[key]], dim="leftover"
            )
            drain_leftover_data[key] = concatenated.mean(dim="leftover")
    else:
        drain_leftover_data = infiltration_drn_data

    return Drainage(**drain_leftover_data)  # type: ignore[arg-type]


class River(TopSystemBoundaryCondition, IRegridPackage):
    """
    River package.
    Any number of RIV Packages can be specified for a single groundwater flow
    model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=71

    Parameters
    ----------
    stage: array of floats (xr.DataArray)
        is the head in the river.
    conductance: array of floats (xr.DataArray)
        is the riverbed hydraulic conductance.
    bottom_elevation: array of floats (xr.DataArray)
        is the elevation of the bottom of the riverbed.
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport, then this array is used
        as the  concentration for inflow over this boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport, then this keyword specifies
        how outflow over this boundary is computed.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of river information will be written
        to the listing file immediately after it is read. Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of river flow rates will be printed to the
        listing file for every stress period time step in which "BUDGET PRINT"
        is specified in Output Control. If there is no Output Control option and
        PRINT FLOWS is specified, then flow rates are printed for the last time
        step of each stress period. Default is False.
    save_flows: ({True, False}, optional)
        Indicates that river flow terms will be written to the file specified
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

    _pkg_id = "riv"
    _period_data = ("stage", "conductance", "bottom_elevation")
    _keyword_map = {}

    _init_schemata = {
        "stage": [
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
        "bottom_elevation": [
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
        "print_input": [DTypeSchema(np.bool_), DimsSchema()],
        "print_flows": [DTypeSchema(np.bool_), DimsSchema()],
        "save_flows": [DTypeSchema(np.bool_), DimsSchema()],
    }
    _write_schemata = {
        "stage": [
            AllValueSchema(">=", "bottom_elevation"),
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "conductance": [IdentityNoDataSchema("stage"), AllValueSchema(">", 0.0)],
        "bottom_elevation": [
            IdentityNoDataSchema("stage"),
            # Check river bottom above layer bottom, else Modflow throws error.
            AllValueSchema(">=", "bottom"),
        ],
        "concentration": [IdentityNoDataSchema("stage"), AllValueSchema(">=", 0.0)],
    }

    _template = TopSystemBoundaryCondition._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}
    _regrid_method = RiverRegridMethod()
    _aggregate_method: DataclassType = RiverAggregationMethod()

    @init_log_decorator()
    def __init__(
        self,
        stage,
        conductance,
        bottom_elevation,
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
            "stage": stage,
            "conductance": conductance,
            "bottom_elevation": bottom_elevation,
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
        kwargs["stage"] = self["stage"]
        kwargs["bottom_elevation"] = self["bottom_elevation"]
        errors = super()._validate(schemata, **kwargs)

        return errors

    @standard_log_decorator()
    def cleanup(self, dis: StructuredDiscretization | VerticesDiscretization) -> None:
        """
        Clean up package inplace. This method calls
        :func:`imod.prepare.cleanup_riv`, see documentation of that
        function for details on cleanup.

        dis: imod.mf6.StructuredDiscretization | imod.mf6.VerticesDiscretization
            Model discretization package.
        """
        dis_dict = {"idomain": dis.dataset["idomain"], "bottom": dis.dataset["bottom"]}
        cleaned_dict = self._call_func_on_grids(cleanup_riv, dis_dict)
        super().__init__(cleaned_dict)

    @classmethod
    def _allocate_and_distribute_planar_data(
        cls,
        planar_data: GridDataDict,
        dis: StructuredDiscretization | VerticesDiscretization,
        npf: NodePropertyFlow,
        allocation_option: ALLOCATION_OPTION,
        distributing_option: DISTRIBUTING_OPTION,
    ) -> tuple[GridDataDict, GridDataDict]:
        """
        Allocate and distribute planar data for given discretization and npf
        package. If layer number of ``planar_data`` is negative,
        ``allocation_option`` is overrided and set to
        ALLOCATION_OPTION.at_first_active.

        Parameters
        ----------
        planar_data: GridDataDict
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
        GridDataDict
            Dictionary with layered grid data.
        """
        top = dis.dataset["top"]
        bottom = dis.dataset["bottom"]
        idomain = dis.dataset["idomain"]

        if has_negative_layer(planar_data["stage"]):
            allocation_option = ALLOCATION_OPTION.at_first_active

        # Enforce planar data, remove all layer dimension information
        planar_data = {
            key: grid.isel({"layer": 0}, drop=True, missing_dims="ignore")
            for key, grid in planar_data.items()
        }
        # Allocation of cells
        riv_allocated, drn_allocated = allocate_riv_cells(
            allocation_option,
            idomain > 0,
            top,
            bottom,
            planar_data["stage"],
            planar_data["bottom_elevation"],
        )
        drn_is_allocated = drn_allocated is not None
        # Distribution of conductances
        allocated_for_distribution = (
            riv_allocated | drn_allocated if drn_is_allocated else riv_allocated  # type: ignore
        )
        distribute_func = (
            distribute_drn_conductance
            if drn_is_allocated
            else distribute_riv_conductance
        )
        distribute_args = (
            distributing_option,
            allocated_for_distribution,
            planar_data["conductance"],
            top,
            bottom,
            npf.dataset["k"],
        )
        riv_distribute_grids = (planar_data["stage"], planar_data["bottom_elevation"])
        drn_distribute_grids = (planar_data["bottom_elevation"],)
        bc_distribute_grids = (
            drn_distribute_grids if drn_is_allocated else riv_distribute_grids
        )
        conductance = distribute_func(*distribute_args, *bc_distribute_grids)
        # Create layered data dicts
        layered_data_riv = {}
        # create layered arrays of stage and bottom elevation
        for key in ["stage", "bottom_elevation"]:
            layered_data_riv[key] = enforce_dim_order(
                planar_data[key].where(riv_allocated)
            )
        layered_data_riv["conductance"] = conductance.where(riv_allocated)

        layered_data_drn = {}
        if drn_allocated is not None:
            layered_data_drn["elevation"] = enforce_dim_order(
                planar_data["stage"].where(drn_allocated)
            )
            layered_data_drn["conductance"] = conductance.where(drn_allocated)

        layered_data_riv["bottom_elevation"] = rise_bottom_elevation_if_needed(
            layered_data_riv["bottom_elevation"], bottom
        )

        return layered_data_riv, layered_data_drn

    @classmethod
    def from_imod5_data(
        cls,
        key: str,
        imod5_data: dict[str, GridDataDict],
        period_data: dict[str, list[datetime]],
        target_dis: StructuredDiscretization,
        target_npf: NodePropertyFlow,
        time_min: datetime,
        time_max: datetime,
        allocation_option: ALLOCATION_OPTION,
        distributing_option: DISTRIBUTING_OPTION,
        regridder_types: Optional[RiverRegridMethod] = None,
        regrid_cache: RegridderWeightsCache = RegridderWeightsCache(),
    ) -> Tuple[Optional["River"], Optional[Drainage]]:
        """
        Construct a river-package from iMOD5 data, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        .. note::

            The method expects the iMOD5 model to be fully 3D, not quasi-3D.

        Parameters
        ----------
        key: str
            Packagename of the package that needs to be converted to river
            package.
        imod5_data: dict
            Dictionary with iMOD5 data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        period_data: dict
            Dictionary with iMOD5 period data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        target_dis:  StructuredDiscretization package
            The grid that should be used for the new package. Does not
            need to be identical to one of the input grids.
        time_min: datetime
            Begin-time of the simulation. Used for expanding period data.
        time_max: datetime
            End-time of the simulation. Used for expanding period data.
        allocation_option: ALLOCATION_OPTION
            allocation option. If package data is assigned to a negative layer
            number, this option is overridden and set to
            ALLOCATION_OPTION.at_first_active.
        distributing_option: DISTRIBUTING_OPTION
            distributing option.
        regridder_types: RiverRegridMethod, optional
            Optional dataclass with regridder types for a specific variable.
            Use this to override default regridding methods.
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to speed up regridding,
            if the same regridders are used several times for regridding different arrays.

        Returns
        -------
        A tuple containing a River package and a Drainage package. The Drainage
        package accounts for the infiltration factor which exists in iMOD5 but
        not in MF6. It furthermore potentially contains drainage cells above
        river stage if ``ALLOCATION_OPTION.stage_to_riv_bot_drn_above`` is
        chosen. Both the river package and the drainage package can be None,
        this can happen if the infiltration factor is 0 or 1 everywhere.
        """
        # gather input data
        varnames = ["conductance", "stage", "bottom_elevation", "infiltration_factor"]
        data = {varname: imod5_data[key][varname] for varname in varnames}
        mask = data["conductance"] > 0
        data["conductance"] = data["conductance"].where(mask)
        # Regrid the input data
        regridded_riv_pkg_data = regrid_imod5_pkg_data(
            cls, data, target_dis, regridder_types, regrid_cache
        )
        regridded_riv_pkg_data = broadcast_and_mask_arrays(regridded_riv_pkg_data)
        # Pop infiltration_factor to avoid unnecessarily allocating and
        # distributing it.
        infiltration_factor = regridded_riv_pkg_data.pop("infiltration_factor")
        # Allocate and distribute planar data if the grid is planar
        is_planar_xy = is_planar_grid(regridded_riv_pkg_data["conductance"])
        allocation_drn_data: GridDataDict = {}
        if is_planar_xy:
            # allocate and distribute planar data
            allocation_riv_data, allocation_drn_data = (
                cls._allocate_and_distribute_planar_data(
                    regridded_riv_pkg_data,
                    target_dis,
                    target_npf,
                    allocation_option,
                    distributing_option,
                )
            )
            regridded_riv_pkg_data.update(allocation_riv_data)
            infiltration_factor = infiltration_factor.isel(
                {"layer": 0}, drop=True, missing_dims="ignore"
            )
        regridded_riv_pkg_data["bottom_elevation"] = enforce_dim_order(
            regridded_riv_pkg_data["bottom_elevation"]
        )
        # Create packages
        regridded_riv_pkg_data, infiltration_drn_data = _separate_infiltration_data(
            regridded_riv_pkg_data, infiltration_factor
        )
        riv_pkg = cls(**regridded_riv_pkg_data, validate=True)
        drn_pkg = _create_drain_from_leftover_riv_imod5_data(
            allocation_drn_data,
            infiltration_drn_data,
        )
        # Mask the river and drainage packages to drop empty data.
        optional_riv_pkg = mask_package__drop_if_empty(riv_pkg)
        optional_drn_pkg = mask_package__drop_if_empty(drn_pkg)

        # Account for periods with repeat stresses.
        repeat = period_data.get(key)
        set_repeat_stress_if_available(repeat, time_min, time_max, optional_riv_pkg)
        set_repeat_stress_if_available(repeat, time_min, time_max, optional_drn_pkg)
        # Clip the river package to the time range of the simulation and ensure
        # time is forward filled.
        optional_riv_pkg = clip_time_if_package(optional_riv_pkg, time_min, time_max)
        optional_drn_pkg = clip_time_if_package(optional_drn_pkg, time_min, time_max)

        # Cast for mypy checks
        optional_riv_pkg = cast(Optional[River], optional_riv_pkg)
        optional_drn_pkg = cast(Optional[Drainage], optional_drn_pkg)

        return (optional_riv_pkg, optional_drn_pkg)
