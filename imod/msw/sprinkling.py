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
from imod.msw.utilities.common import concat_imod5
from imod.typing import Imod5DataDict, IntArray
from imod.typing.grid import zeros_like


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
        "svat_groundwater": VariableMetaData(10, None, None, str),
        "layer": VariableMetaData(6, 1, 9999, int),
        "trajectory": VariableMetaData(10, None, None, str),
    }

    _with_subunit = ()
    _without_subunit = (
        "max_abstraction_groundwater",
        "max_abstraction_surfacewater",
    )

    _to_fill = (
        "max_abstraction_groundwater_mm_d",
        "max_abstraction_surfacewater_mm_d",
        "svat_groundwater",
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
        if len(well_cellid.coords["dim_cellid"]) != 3:
            raise TypeError("Coupling to unstructured grids is not supported.")

        well_layer = well_cellid.sel(dim_cellid="layer").data
        well_row = well_cellid.sel(dim_cellid="row").data - 1
        well_column = well_cellid.sel(dim_cellid="column").data - 1

        n_subunit = svat["subunit"].size

        well_svat = svat.data[:, well_row, well_column]
        well_active = well_svat != 0

        # Tile well_layers for each subunit
        layer = np.tile(well_layer, (n_subunit, 1))

        data_dict = {"svat": well_svat[well_active], "layer": layer[well_active]}

        for var in self._without_subunit:
            well_arr = self.dataset[var].data[well_row, well_column]
            well_arr = np.tile(well_arr, (n_subunit, 1))
            data_dict[var] = well_arr[well_active]

        for var in self._to_fill:
            data_dict[var] = ""

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict) -> "Sprinkling":
        cap_data = imod5_data["cap"]
        if isinstance(cap_data["artificial_recharge_layer"], pd.DataFrame):
            raise NotImplementedError(
                "Assigning sprinkling wells with an IPF file is not supported, please specify them as IDF."
            )
        drop_layer_kwargs = {
            "layer": 0,
            "drop": True,
            "missing_dims": "ignore",
        }
        type = cap_data["artificial_recharge"].isel(**drop_layer_kwargs)
        capacity = cap_data["artificial_recharge_capacity"].isel(**drop_layer_kwargs)
        max_abstraction_groundwater_rural = capacity.where(type == 1)
        max_abstraction_surfacewater_rural = capacity.where(type == 2).fillna(0.0)

        max_abstraction_urban = zeros_like(type)

        data = {}
        data["max_abstraction_groundwater"] = concat_imod5(
            max_abstraction_groundwater_rural, max_abstraction_urban
        )
        data["max_abstraction_surfacewater"] = concat_imod5(
            max_abstraction_surfacewater_rural, max_abstraction_urban
        )

        return cls(**data)
