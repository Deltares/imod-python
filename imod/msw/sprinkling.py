from typing import TextIO

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.dis import StructuredDiscretization
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.regrid.regrid_schemes import SprinklingRegridMethod
from imod.typing import IntArray


def _ravel_per_subunit(da: xr.DataArray) -> np.ndarray:
    # per defined well element, all subunits
    array_out = da.to_numpy().ravel()
    # per defined well element, per defined subunits
    return array_out[np.isfinite(array_out)]


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
