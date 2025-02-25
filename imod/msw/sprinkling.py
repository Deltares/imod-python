from typing import TextIO

import numpy as np
import pandas as pd
import xarray as xr

from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.regrid.regrid_schemes import SprinklingRegridMethod
from imod.msw.utilities.common import concat_imod5
from imod.msw.utilities.imod5_converter import (
    get_cell_area_from_imod5_data,
)
from imod.typing import GridDataDict, Imod5DataDict, IntArray
from imod.typing.grid import zeros_like


def _ravel_per_subunit(da: xr.DataArray) -> np.ndarray:
    # per defined well element, all subunits
    array_out = da.to_numpy().ravel()
    # per defined well element, per defined subunits
    return array_out[np.isfinite(array_out)]


def _sprinkling_data_from_imod5_ipf(cap_data: GridDataDict) -> GridDataDict:
    raise NotImplementedError(
        "Assigning sprinkling wells with an IPF file is not supported, please specify them as IDF."
    )


def _sprinkling_data_from_imod5_grid(cap_data: GridDataDict) -> GridDataDict:
    # Convert units from mm/d to m3/d
    msw_area = get_cell_area_from_imod5_data(cap_data)
    capacity_mmd = cap_data["artificial_recharge_capacity"]
    capacity_m3d = capacity_mmd * 1e-3 * msw_area.sel(subunit=0, drop=True)

    artificial_rch_type = cap_data["artificial_recharge"]
    from_groundwater = artificial_rch_type == 1
    from_surfacewater = artificial_rch_type == 2
    is_active = artificial_rch_type != 0

    zero_where_active = zeros_like(artificial_rch_type).where(is_active)

    # Add zero where active, to have active cells set to 0.0.
    max_abstraction_groundwater_rural = zero_where_active.where(
        ~from_groundwater, capacity_m3d
    )
    max_abstraction_surfacewater_rural = zero_where_active.where(
        ~from_surfacewater, capacity_m3d
    )

    # No sprinkling for urban environments
    max_abstraction_urban = zero_where_active

    data = {}
    data["max_abstraction_groundwater"] = concat_imod5(
        max_abstraction_groundwater_rural, max_abstraction_urban
    )
    data["max_abstraction_surfacewater"] = concat_imod5(
        max_abstraction_surfacewater_rural, max_abstraction_urban
    )
    return data


class Sprinkling(MetaSwapPackage, IRegridPackage):
    """
    This contains the sprinkling capacities of links between SVAT units and
    groundwater/surface water locations.

    This class is responsible for the file `scap_svat.inp`

    Parameters
    ----------
    max_abstraction_groundwater: array of floats (xr.DataArray)
        Describes the maximum abstraction of groundwater to SVAT units in m3 per
        day. This array must not have a subunit coordinate.
    max_abstraction_surfacewater: array of floats (xr.DataArray)
        Describes the maximum abstraction of surfacewater to SVAT units in m3
        per day. This array must not have a subunit coordinate.
    """

    _file_name = "scap_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "max_abstraction_groundwater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_surfacewater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_groundwater": VariableMetaData(8, 0.0, 1e9, float),
        "max_abstraction_surfacewater": VariableMetaData(8, 0.0, 1e9, float),
        "svat_groundwater": VariableMetaData(10, 1, 99999999, int),
        "layer": VariableMetaData(6, 1, 9999, int),
        "trajectory": VariableMetaData(10, None, None, str),
    }

    _with_subunit = (
        "max_abstraction_groundwater",
        "max_abstraction_surfacewater",
    )
    _without_subunit = ()

    _to_fill = (
        "max_abstraction_groundwater_mm_d",
        "max_abstraction_surfacewater_mm_d",
        "trajectory",
    )

    _regrid_method = SprinklingRegridMethod()

    def __init__(
        self,
        max_abstraction_groundwater: xr.DataArray,
        max_abstraction_surfacewater: xr.DataArray,
    ):
        super().__init__()
        self.dataset["max_abstraction_groundwater"] = max_abstraction_groundwater
        self.dataset["max_abstraction_surfacewater"] = max_abstraction_surfacewater

        self._pkgcheck()

    def _render(
        self,
        file: TextIO,
        index: IntArray,
        svat: xr.DataArray,
        mf6_dis: StructuredDiscretization,
        mf6_well: Mf6Wel,
    ):
        if not isinstance(mf6_well, Mf6Wel):
            raise TypeError(rf"well not of type 'Mf6Wel', got '{type(mf6_well)}'")

        well_cellid = mf6_well["cellid"]

        well_layer = well_cellid.sel(dim_cellid="layer").data
        well_row = well_cellid.sel(dim_cellid="row").data - 1
        well_column = well_cellid.sel(dim_cellid="column").data - 1

        max_rate_per_svat = self.dataset["max_abstraction_groundwater"].where(svat > 0)
        well_layer_per_svat = xr.full_like(max_rate_per_svat, np.nan)
        well_layer_per_svat.values[:, well_row, well_column] = well_layer

        is_active_per_svat = (max_rate_per_svat > 0) & well_layer_per_svat.notnull()

        layer_active = well_layer_per_svat.where(is_active_per_svat)
        layer_source = _ravel_per_subunit(layer_active).astype(dtype=np.int32)
        svat_active = svat.where(is_active_per_svat)
        svat_source_target = _ravel_per_subunit(svat_active).astype(dtype=np.int32)

        data_dict: dict[str, str | np.ndarray] = {
            "svat": svat_source_target,
            "layer": layer_source,
            "svat_groundwater": svat_source_target,
        }

        for var in self._with_subunit:
            data_with_well = self.dataset[var].where(is_active_per_svat)
            data_dict[var] = _ravel_per_subunit(data_with_well)

        for var in self._to_fill:
            data_dict[var] = ""

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict) -> "Sprinkling":
        """
        Import sprinkling data from imod5 data. Abstraction data for sprinkling
        is defined in iMOD5 either with grids (IDF) or points (IPF) combined
        with a grid. Depending on the type, the method does different conversions:

        - grids (IDF)
            The ``"artifical_recharge_layer"`` variable was defined as grid
            (IDF), this grid defines in which layer a groundwater abstraction
            well should be placed. The ``"artificial_recharge"`` grid contains
            types which point to the type of abstraction:

                * 0: no abstraction
                * 1: groundwater abstraction
                * 2: surfacewater abstraction

            The ``"artificial_recharge_capacity"`` grid/constant defines the
            capacity of each groundwater or surfacewater abstraction. This is an
            ``1:1`` mapping: Each grid cell maps to a separate well.

        - points with grid (IPF & IDF)
            The ``"artifical_recharge_layer"`` variable was defined as point
            data (IPF), this table contains wellids with an abstraction capacity
            and layer. The ``"artificial_recharge"`` grid contains a mapping of
            grid cells to wellids in the point data. The
            ``"artificial_recharge_capacity"`` is ignored as the abstraction
            capacity is already defined in the point data. This is an ``n:1``
            mapping: multiple grid cells can map to one well.

        Parameters
        ----------
        imod5_data: dict[str, dict[str, GridDataArray]]
            dictionary containing the arrays mentioned in the project file as
            xarray datasets, under the key of the package type to which it
            belongs, as returned by
            :func:`imod.formats.prj.open_projectfile_data`.

        Returns
        -------
        Sprinkling package
        """
        cap_data = imod5_data["cap"]
        if isinstance(cap_data["artificial_recharge_layer"], pd.DataFrame):
            data = _sprinkling_data_from_imod5_ipf(cap_data)
        else:
            data = _sprinkling_data_from_imod5_grid(cap_data)

        return cls(**data)
