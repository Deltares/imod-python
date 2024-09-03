from copy import deepcopy
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import xarray as xr

from imod import logging
from imod.logging import init_log_decorator
from imod.logging.loglevel import LogLevel
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.disv import VerticesDiscretization
from imod.mf6.drn import Drainage
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.regrid.regrid_schemes import RiverRegridMethod
from imod.mf6.utilities.regrid import (
    RegridderWeightsCache,
    _regrid_package_data,
)
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA, CONC_DIMS_SCHEMA
from imod.prepare.cleanup import cleanup_riv
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION, allocate_riv_cells
from imod.prepare.topsystem.conductance import (
    DISTRIBUTING_OPTION,
    distribute_riv_conductance,
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


class River(BoundaryCondition, IRegridPackage):
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
    repeat_stress: Optional[xr.DataArray] of datetimes
        Used to repeat data for e.g. repeating stress periods such as
        seasonality without duplicating the values. The DataArray should have
        dimensions ``("repeat", "repeat_items")``. The ``repeat_items``
        dimension should have size 2: the first value is the "key", the second
        value is the "value". For the "key" datetime, the data of the "value"
        datetime will be used. Can also be set with a dictionary using the
        ``set_repeat_stress`` method.
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
        ],
        "conductance": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
        ],
        "bottom_elevation": [
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

    _template = BoundaryCondition._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}
    _regrid_method = RiverRegridMethod()

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

    def cleanup(self, dis: StructuredDiscretization | VerticesDiscretization) -> None:
        """
        Clean up package inplace. This method calls
        :func:`imod.prepare.cleanup.cleanup_riv`, see documentation of that
        function for details on cleanup.

        dis: imod.mf6.StructuredDiscretization | imod.mf6.VerticesDiscretization
            Model discretization package.
        """
        dis_dict = {"idomain": dis.dataset["idomain"], "bottom": dis.dataset["bottom"]}
        cleaned_dict = self._call_func_on_grids(cleanup_riv, dis_dict)
        super().__init__(cleaned_dict)

    @classmethod
    def from_imod5_data(
        cls,
        key: str,
        imod5_data: dict[str, dict[str, GridDataArray]],
        period_data: dict[str, list[datetime]],
        target_discretization: StructuredDiscretization,
        time_min: datetime,
        time_max: datetime,
        allocation_option_riv: ALLOCATION_OPTION,
        distributing_option_riv: DISTRIBUTING_OPTION,
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
        target_discretization:  StructuredDiscretization package
            The grid that should be used for the new package. Does not
            need to be identical to one of the input grids.
        time_min: datetime
            Begin-time of the simulation. Used for expanding period data.
        time_max: datetime
            End-time of the simulation. Used for expanding period data.
        allocation_option: ALLOCATION_OPTION
            allocation option.
        distributing_option: dict[str, DISTRIBUTING_OPTION]
            distributing option.
        regridder_types: RiverRegridMethod, optional
            Optional dataclass with regridder types for a specific variable.
            Use this to override default regridding methods.
        regrid_cache:Optional RegridderWeightsCache
            stores regridder weights for different regridders. Can be used to speed up regridding,
            if the same regridders are used several times for regridding different arrays.

        Returns
        -------
        A MF6 river package, and a drainage package to account
        for the infiltration factor which exists in IMOD5 but not in MF6.
        Both the river package and the drainage package can be None,
        this can happen if the infiltration factor is 0 or 1 everywhere.
        """

        logger = logging.logger
        # gather discretrizations
        target_top = target_discretization.dataset["top"]
        target_bottom = target_discretization.dataset["bottom"]
        target_idomain = target_discretization.dataset["idomain"]

        # gather input data
        data = {
            "conductance": imod5_data[key]["conductance"].copy(deep=True),
            "stage": imod5_data[key]["stage"].copy(deep=True),
            "bottom_elevation": imod5_data[key]["bottom_elevation"].copy(deep=True),
            "infiltration_factor": imod5_data[key]["infiltration_factor"].copy(
                deep=True
            ),
        }
        is_planar_conductance = is_planar_grid(data["conductance"])

        # set up regridder methods
        if regridder_types is None:
            regridder_types = River.get_regrid_methods()
        # regrid the input data
        regridded_package_data = _regrid_package_data(
            data, target_idomain, regridder_types, regrid_cache, {}
        )

        conductance = regridded_package_data["conductance"]
        infiltration_factor = regridded_package_data["infiltration_factor"]

        if is_planar_conductance:
            riv_allocation = allocate_riv_cells(
                allocation_option_riv,
                target_idomain == 1,
                target_top,
                target_bottom,
                regridded_package_data["stage"],
                regridded_package_data["bottom_elevation"],
            )

            regridded_package_data["conductance"] = distribute_riv_conductance(
                distributing_option_riv,
                riv_allocation[0],
                conductance,
                target_top,
                target_bottom,
                conductance,
                regridded_package_data["stage"],
                regridded_package_data["bottom_elevation"],
            )

            # create layered arrays of stage and bottom elevation
            layered_stage = regridded_package_data["stage"].where(riv_allocation[0])
            layered_stage = enforce_dim_order(layered_stage)
            regridded_package_data["stage"] = layered_stage

            layered_bottom_elevation = regridded_package_data["bottom_elevation"].where(
                riv_allocation[0]
            )
            layered_bottom_elevation = enforce_dim_order(layered_bottom_elevation)

            # due to regridding, the layered_bottom_elevation could be smaller than the
            # bottom, so here we overwrite it with bottom if that's
            # the case.

            if np.any((target_bottom > layered_bottom_elevation).values[()]):
                logger.log(
                    loglevel=LogLevel.WARNING,
                    message="Note: riv bottom was detected below model bottom. Updated the riv's bottom.",
                    additional_depth=0,
                )
            layered_bottom_elevation = xr.where(
                target_bottom > layered_bottom_elevation,
                target_bottom,
                layered_bottom_elevation,
            )

            regridded_package_data["bottom_elevation"] = layered_bottom_elevation

        # update the conductance of the river package to account for the infiltration
        # factor
        drain_conductance, river_conductance = cls.split_conductance(
            regridded_package_data["conductance"], infiltration_factor
        )
        regridded_package_data["conductance"] = river_conductance
        regridded_package_data.pop("infiltration_factor")
        regridded_package_data["bottom_elevation"] = enforce_dim_order(
            regridded_package_data["bottom_elevation"]
        )

        river_package = River(**regridded_package_data, validate=True)
        optional_river_package: Optional[River] = None
        optional_drainage_package: Optional[Drainage] = None
        # create a drainage package with the conductance we computed from the infiltration factor
        drainage_arrays = {
            "stage": regridded_package_data["stage"],
            "conductance": drain_conductance,
        }

        drainage_package = cls.create_infiltration_factor_drain(
            drainage_arrays["stage"],
            drainage_arrays["conductance"],
        )
        # remove River package if its mask is False everywhere
        mask = ~np.isnan(river_conductance)
        if np.any(mask):
            optional_river_package = river_package.mask(mask)
        else:
            optional_river_package = None

        # remove Drainage package if its mask is False everywhere
        mask = ~np.isnan(drain_conductance)
        if np.any(mask):
            optional_drainage_package = drainage_package.mask(mask)
        else:
            optional_drainage_package = None

        repeat = period_data.get(key)
        if repeat is not None:
            if optional_river_package is not None:
                optional_river_package.set_repeat_stress(
                    expand_repetitions(repeat, time_min, time_max)
                )
            if optional_drainage_package is not None:
                optional_drainage_package.set_repeat_stress(
                    expand_repetitions(repeat, time_min, time_max)
                )

        return (optional_river_package, optional_drainage_package)

    @classmethod
    def create_infiltration_factor_drain(
        cls,
        drain_elevation: GridDataArray,
        drain_conductance: GridDataArray,
    ):
        """
        Create a drainage package from the river package, to account for the infiltration factor.
        This factor is optional in imod5, but it does not exist in MF6, so we mimic its effect
        with a Drainage boundary.
        """

        mask = ~np.isnan(drain_conductance)
        drainage = Drainage(drain_elevation, drain_conductance)
        drainage.mask(mask)
        return drainage

    @classmethod
    def split_conductance(cls, conductance, infiltration_factor):
        """
        Seperates (exfiltration) conductance with an infiltration factor (iMODFLOW) into
        a drainage conductance and a river conductance following methods explained in Zaadnoordijk (2009).

        Parameters
        ----------
        conductance : xr.DataArray or float
            Exfiltration conductance. Is the default conductance provided to the iMODFLOW river package
        infiltration_factor : xr.DataArray or float
            Infiltration factor. The exfiltration conductance is multiplied with this factor to compute
            the infiltration conductance. If 0, no infiltration takes place; if 1, infiltration is equal to    exfiltration

        Returns
        -------
        drainage_conductance : xr.DataArray
            conductance for the drainage package
        river_conductance : xr.DataArray
            conductance for the river package

        Derivation
        ----------
        From Zaadnoordijk (2009):
        [1] cond_RIV = A/ci
        [2] cond_DRN = A * (ci-cd) / (ci*cd)
        Where cond_RIV and cond_DRN repsectively are the River and Drainage conductance [L^2/T],
        A is the cell area [L^2] and ci and cd respectively are the infiltration and exfiltration resistance [T]

        Taking f as the infiltration factor and cond_d as the exfiltration conductance, we can write (iMOD manual):
        [3] ci = cd * (1/f)
        [4] cond_d = A/cd

        We can then rewrite equations 1 and 2 to:
        [5] cond_RIV = f * cond_d
        [6] cond_DRN = (1-f) * cond_d

        References
        ----------
        Zaadnoordijk, W. (2009).
        Simulating Piecewise-Linear Surface Water and Ground Water Interactions with MODFLOW.
        Ground Water.
        https://ngwa.onlinelibrary.wiley.com/doi/10.1111/j.1745-6584.2009.00582.x

        iMOD manual v5.2 (2020)
        https://oss.deltares.nl/web/imod/

        """
        if np.any(infiltration_factor > 1):
            raise ValueError("The infiltration factor should not exceed 1")

        drainage_conductance = conductance * (1 - infiltration_factor)

        river_conductance = conductance * infiltration_factor

        # clean up the packages
        drainage_conductance = drainage_conductance.where(drainage_conductance > 0)
        river_conductance = river_conductance.where(river_conductance > 0)
        return drainage_conductance, river_conductance

    @classmethod
    def get_regrid_methods(cls) -> RiverRegridMethod:
        return deepcopy(cls._regrid_method)
