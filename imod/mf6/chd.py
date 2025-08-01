from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np

from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.logging import init_log_decorator
from imod.logging.logging_decorators import standard_log_decorator
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.regrid.regrid_schemes import ConstantHeadRegridMethod
from imod.mf6.utilities.imod5_converter import (
    chd_cells_from_imod5_data,
    regrid_imod5_pkg_data,
)
from imod.mf6.utilities.package import set_repeat_stress_if_available
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA, CONC_DIMS_SCHEMA
from imod.schemata import (
    AllCoordsValueSchema,
    AllInsideNoDataSchema,
    AllNoDataSchema,
    AllValueSchema,
    CoordsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
    OtherCoordsSchema,
)
from imod.typing import GridDataArray
from imod.util.regrid import RegridderWeightsCache


class ConstantHead(BoundaryCondition, IRegridPackage):
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

    _pkg_id = "chd"
    _keyword_map = {}
    _period_data = ("head",)

    _init_schemata = {
        "head": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "concentration": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            CONC_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
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
    _regrid_method = ConstantHeadRegridMethod()

    @init_log_decorator()
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
        dict_dataset = {
            "head": head,
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

    @classmethod
    @standard_log_decorator()
    def from_imod5_data(
        cls,
        key: str,
        imod5_data: dict[str, dict[str, GridDataArray]],
        period_data: dict[str, list[datetime]],
        target_dis: StructuredDiscretization,
        time_min: datetime,
        time_max: datetime,
        regridder_types: Optional[ConstantHeadRegridMethod] = None,
        regrid_cache: RegridderWeightsCache = RegridderWeightsCache(),
    ) -> "ConstantHead":
        """
        Construct a ConstantHead-package from iMOD5 data, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        This function can be used if chd packages are defined in the imod5 data.

        If they are not, then imod5 assumed that at all the locations where ibound
        = -1 a chd package is active with the starting head of the simulation
        as a constant. In that case, use the from_imod5_shd_data function instead
        of this one.

        The creation of a chd package from shd data should only be done if no chd
        packages at all are present in the imod5_data

        Parameters
        ----------
        key: str
            The key used in the imod5 data dictionary that is used to refer
            to the chd package that we want to import.
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
        regridder_types: RegridMethodType, optional
            Optional dataclass with regridder types for a specific variable.
            Use this to override default regridding methods.
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to speed up regridding,
            if the same regridders are used several times for regridding different arrays.

        Returns
        -------
        A list of Modflow 6 ConstantHead packages.
        """
        return cls._from_head_data(
            key,
            imod5_data[key]["head"],
            imod5_data["bnd"]["ibound"],
            period_data,
            target_dis,
            time_min,
            time_max,
            regridder_types,
            regrid_cache,
        )

    @classmethod
    @standard_log_decorator()
    def from_imod5_shd_data(
        cls,
        imod5_data: dict[str, dict[str, GridDataArray]],
        period_data: dict[str, list[datetime]],
        target_dis: StructuredDiscretization,
        regridder_types: Optional[ConstantHeadRegridMethod] = None,
        regrid_cache: RegridderWeightsCache = RegridderWeightsCache(),
    ) -> "ConstantHead":
        """
        Construct a ConstantHead-package from iMOD5 data, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        This function can be used if no chd packages at all are defined in the imod5 data.

        In that case, imod5 assumed that at all the locations where ibound
        = -1,  a chd package is active with the starting head of the simulation
        as a constant.

        So this function creates a single chd package that will be present at all locations where
        ibound == -1. The assigned head will be the starting head, specified in the array "shd"
        in the imod5 data.

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
        regridder_types: ConstantHeadRegridMethod, optional
            Optional dataclass with regridder types for a specific variable.
            Use this to override default regridding methods.
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to speed up regridding,
            if the same regridders are used several times for regridding different arrays.

        Returns
        -------
        A  Modflow 6 ConstantHead package.
        """
        return cls._from_head_data(
            "shd",
            imod5_data["shd"]["head"],
            imod5_data["bnd"]["ibound"],
            period_data,
            target_dis,
            regridder_types=regridder_types,
            regrid_cache=regrid_cache,
        )

    @classmethod
    def _from_head_data(
        cls,
        key: str,
        head: GridDataArray,
        ibound: GridDataArray,
        period_data: dict[str, list[datetime]],
        target_dis: StructuredDiscretization,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        regridder_types: Optional[ConstantHeadRegridMethod] = None,
        regrid_cache: RegridderWeightsCache = RegridderWeightsCache(),
    ) -> "ConstantHead":
        target_idomain = target_dis.dataset["idomain"]

        data = {"head": head, "ibound": ibound}

        regridded_pkg_data = regrid_imod5_pkg_data(
            cls, data, target_dis, regridder_types, regrid_cache
        )
        chd_pkg_data = chd_cells_from_imod5_data(regridded_pkg_data, target_idomain)
        chd = cls(**chd_pkg_data, validate=True)
        if time_min is not None and time_max is not None:
            repeat = period_data.get(key)
            set_repeat_stress_if_available(repeat, time_min, time_max, chd)
            # Clip the drain package to the time range of the simulation and ensure
            # time is forward filled.
            chd = chd.clip_box(time_min=time_min, time_max=time_max)

        return chd

    @classmethod
    def get_regrid_methods(cls) -> ConstantHeadRegridMethod:
        return deepcopy(cls._regrid_method)
