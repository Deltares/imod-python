from copy import deepcopy
from typing import Optional, cast

import numpy as np
import xarray as xr

from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.common.utilities.regrid import _regrid_package_data
from imod.logging import init_log_decorator
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.regrid.regrid_schemes import RechargeRegridMethod
from imod.mf6.utilities.imod5_converter import convert_unit_rch_rate
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA, CONC_DIMS_SCHEMA
from imod.msw.utilities.imod5_converter import (
    get_cell_area_from_imod5_data,
    is_msw_active_cell,
)
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION, allocate_rch_cells
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
from imod.typing import GridDataArray, GridDataDict, Imod5DataDict
from imod.typing.grid import (
    enforce_dim_order,
    is_planar_grid,
)
from imod.util.dims import drop_layer_dim_cap_data
from imod.util.regrid import RegridderWeightsCache


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
        "rate": [
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "concentration": [IdentityNoDataSchema("rate"), AllValueSchema(">=", 0.0)],
    }

    _template = BoundaryCondition._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}
    _regrid_method = RechargeRegridMethod()

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

    @classmethod
    def allocate_planar_data(
        cls,
        planar_data: dict[str, GridDataArray],
        dis: StructuredDiscretization,
    ) -> dict[str, GridDataArray]:
        """
        Allocate and distribute planar data for given discretization and npf
        package. To allocate cells, the allocation option
        ALLOCATION_OPTION.at_first_active is set.

        Parameters
        ----------
        planar_data: dict[str, GridDataArray]
            Dictionary with planar grid data.
        dis: imod.mf6.StructuredDiscretization
            Model discretization package.
        npf: imod.mf6.NodePropertyFlow
            Node property flow package.

        Returns
        -------
        dict[str, GridDataArray]
            Dictionary with layered grid data.
        """
        idomain = dis.dataset["idomain"]
        if "layer" in planar_data["rate"].dims:
            planar_data["rate"] = planar_data["rate"].isel(layer=0, drop=True)
        # create an array indicating in which cells rch is active
        is_rch_cell = allocate_rch_cells(
            ALLOCATION_OPTION.at_first_active,
            idomain > 0,
            planar_data["rate"],
        )
        # remove rch from cells where it is not allocated and broadcast over layers.
        layered_data = {}
        layered_data["rate"] = planar_data["rate"].where(is_rch_cell)
        layered_data["rate"] = enforce_dim_order(layered_data["rate"])
        return layered_data

    @classmethod
    def from_imod5_data(
        cls,
        imod5_data: dict[str, dict[str, GridDataArray]],
        target_dis: StructuredDiscretization,
        regridder_types: Optional[RechargeRegridMethod] = None,
        regrid_cache: RegridderWeightsCache = RegridderWeightsCache(),
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
        target_dis: GridDataArray
            The discretization package for the simulation. Its grid does not
            need to be identical to one of the input grids.
        regridder_types: RechargeRegridMethod, optional
            Optional dataclass with regridder types for a specific variable.
            Use this to override default regridding methods.
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to speed up regridding,
            if the same regridders are used several times for regridding different arrays.

        Returns
        -------
        Modflow 6 rch package.

        """
        new_idomain = target_dis.dataset["idomain"]
        data = {
            "rate": convert_unit_rch_rate(imod5_data["rch"]["rate"]),
        }
        # first regrid the inputs to the target grid.
        if regridder_types is None:
            regridder_settings = Recharge.get_regrid_methods()

        regridded_package_data = _regrid_package_data(
            data, new_idomain, regridder_settings, regrid_cache, {}
        )

        # if rate has only layer 0, then it is planar.
        if is_planar_grid(regridded_package_data["rate"]):
            layered_data = cls.allocate_planar_data(regridded_package_data, target_dis)
            regridded_package_data.update(layered_data)

        return cls(**regridded_package_data, validate=True, fixed_cell=False)

    @classmethod
    def from_imod5_cap_data(
        cls,
        imod5_data: Imod5DataDict,
        target_dis: StructuredDiscretization,
    ) -> "Recharge":
        """
        Construct an rch-package from iMOD5 data in the CAP package, loaded with
        the :func:`imod.formats.prj.open_projectfile_data` function. Package is
        used to couple MODFLOW6 to MetaSWAP models. Active cells will have a
        recharge rate of 0.0.
        """
        cap_data = cast(GridDataDict, drop_layer_dim_cap_data(imod5_data)["cap"])

        msw_area = get_cell_area_from_imod5_data(cap_data)
        msw_active = is_msw_active_cell(target_dis, cap_data, msw_area)
        active = msw_active.all

        data = {}
        zero_scalar = xr.DataArray(0.0, coords={"layer": 1})
        data["rate"] = zero_scalar.where(active)

        return cls(**data, validate=True, fixed_cell=False)

    @classmethod
    def get_regrid_methods(cls) -> RechargeRegridMethod:
        return deepcopy(cls._regrid_method)
