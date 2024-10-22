from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np

from imod.logging import init_log_decorator, standard_log_decorator
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.disv import VerticesDiscretization
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.regrid.regrid_schemes import (
    GeneralHeadBoundaryRegridMethod,
    RegridMethodType,
)
from imod.mf6.utilities.regrid import RegridderWeightsCache, _regrid_package_data
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA, CONC_DIMS_SCHEMA
from imod.prepare.cleanup import cleanup_ghb
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION, allocate_ghb_cells
from imod.prepare.topsystem.conductance import (
    DISTRIBUTING_OPTION,
    distribute_ghb_conductance,
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


class GeneralHeadBoundary(BoundaryCondition, IRegridPackage):
    """
    The General-Head Boundary package is used to simulate head-dependent flux
    boundaries.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=75

    Parameters
    ----------
    head: array of floats (xr.DataArray)
        is the boundary head. (bhead)
    conductance: array of floats (xr.DataArray)
        is the hydraulic conductance of the interface between the aquifer cell and
        the boundary.(cond)
    concentration: array of floats (xr.DataArray, optional)
        if this flow package is used in simulations also involving transport, then this array is used
        as the  concentration for inflow over this boundary.
    concentration_boundary_type: ({"AUX", "AUXMIXED"}, optional)
        if this flow package is used in simulations also involving transport, then this keyword specifies
        how outflow over this boundary is computed.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of general head boundary information
        will be written to the listing file immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of general head boundary flow rates will be
        printed to the listing file for every stress period time step in which
        "BUDGET PRINT" is specified in Output Control. If there is no Output
        Control option and PRINT FLOWS is specified, then flow rates are printed
        for the last time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that general head boundary flow terms will be written to the
        file specified with "BUDGET FILEOUT" in Output Control.
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

    _pkg_id = "ghb"
    _period_data = ("head", "conductance")

    _init_schemata = {
        "head": [
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
        "head": [
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "conductance": [IdentityNoDataSchema("head"), AllValueSchema(">", 0.0)],
        "concentration": [IdentityNoDataSchema("head"), AllValueSchema(">=", 0.0)],
    }

    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}
    _regrid_method = GeneralHeadBoundaryRegridMethod()

    @init_log_decorator()
    def __init__(
        self,
        head,
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
            "head": head,
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
        kwargs["head"] = self["head"]
        errors = super()._validate(schemata, **kwargs)

        return errors

    @standard_log_decorator()
    def cleanup(self, dis: StructuredDiscretization | VerticesDiscretization) -> None:
        """
        Clean up package inplace. This method calls
        :func:`imod.prepare.cleanup_ghb`, see documentation of that
        function for details on cleanup.

        dis: imod.mf6.StructuredDiscretization | imod.mf6.VerticesDiscretization
            Model discretization package.
        """
        dis_dict = {"idomain": dis.dataset["idomain"]}
        cleaned_dict = self._call_func_on_grids(cleanup_ghb, dis_dict)
        super().__init__(cleaned_dict)

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
        regridder_types: Optional[RegridMethodType] = None,
        regrid_cache: RegridderWeightsCache = RegridderWeightsCache(),
    ) -> "GeneralHeadBoundary":
        """
        Construct a GeneralHeadBoundary-package from iMOD5 data, loaded with the
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
        target_dis:  StructuredDiscretization package
            The grid that should be used for the new package. Does not
            need to be identical to one of the input grids.
        target_npf: NodePropertyFlow package
            The conductivity information, used to compute GHB flux
        allocation_option: ALLOCATION_OPTION
            allocation option.
        time_min: datetime
            Begin-time of the simulation. Used for expanding period data.
        time_max: datetime
            End-time of the simulation. Used for expanding period data.
        distributing_option: dict[str, DISTRIBUTING_OPTION]
            distributing option.
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to speed up regridding,
            if the same regridders are used several times for regridding different arrays.
        regridder_types: RegridMethodType, optional
            Optional dataclass with regridder types for a specific variable.
            Use this to override default regridding methods.

        Returns
        -------
        A  Modflow 6 GeneralHeadBoundary packages.
        """
        target_top = target_dis.dataset["top"]
        target_bottom = target_dis.dataset["bottom"]
        target_idomain = target_dis.dataset["idomain"]

        idomain = target_dis.dataset["idomain"]
        data = {
            "head": imod5_data[key]["head"],
            "conductance": imod5_data[key]["conductance"],
        }
        is_planar = is_planar_grid(data["conductance"])

        if regridder_types is None:
            regridder_types = GeneralHeadBoundaryRegridMethod()

        regridded_package_data = _regrid_package_data(
            data, idomain, regridder_types, regrid_cache, {}
        )
        if is_planar:
            conductance = regridded_package_data["conductance"]

            planar_head = regridded_package_data["head"]
            k = target_npf.dataset["k"]

            ghb_allocation = allocate_ghb_cells(
                allocation_option,
                target_idomain == 1,
                target_top,
                target_bottom,
                planar_head,
            )

            layered_head = planar_head.where(ghb_allocation)
            layered_head = enforce_dim_order(layered_head)

            regridded_package_data["head"] = layered_head

            if "layer" in conductance.coords:
                conductance = conductance.isel({"layer": 0}, drop=True)

            regridded_package_data["conductance"] = distribute_ghb_conductance(
                distributing_option,
                ghb_allocation,
                conductance,
                target_top,
                target_bottom,
                k,
            )

        ghb = GeneralHeadBoundary(**regridded_package_data, validate=True)
        repeat = period_data.get(key)
        if repeat is not None:
            ghb.set_repeat_stress(expand_repetitions(repeat, time_min, time_max))
        return ghb

    @classmethod
    def get_regrid_methods(cls) -> GeneralHeadBoundaryRegridMethod:
        return deepcopy(cls._regrid_method)
