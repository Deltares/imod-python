from typing import Any, TextIO

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.dis import StructuredDiscretization
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import DataDictType, MetaSwapPackage
from imod.typing import IntArray


class CouplerMapping(MetaSwapPackage):
    """
    This contains the data to connect MODFLOW 6 cells to MetaSWAP svats.

    This class is responsible for the file `mod2svat.inp`. It also includes
    connection to wells.

    """

    _file_name = "mod2svat.inp"
    _metadata_dict = {
        "mod_id": VariableMetaData(10, 1, 9999999, int),
        "free": VariableMetaData(2, None, None, str),
        "svat": VariableMetaData(10, 1, 9999999, int),
        "layer": VariableMetaData(5, 0, 9999, int),
    }

    _with_subunit = ()
    _without_subunit = ()
    _to_fill = ("free",)

    def __init__(
        self,
    ):
        super().__init__()

    def _create_mod_id_rch(
        self, svat: xr.DataArray, idomain_top_active: xr.DataArray
    ) -> xr.DataArray:
        """
        Create modflow indices for the recharge layer, which is where
        infiltration will take place.
        """
        mod_id = xr.full_like(svat, fill_value=0, dtype=np.int64)
        n_subunit = svat["subunit"].size

        n_mod_top = idomain_top_active.sum()

        # idomain does not have a subunit dimension, so tile for n_subunits
        mod_id_1d: IntArray = np.tile(np.arange(1, n_mod_top + 1), (n_subunit, 1))

        mod_id.data[:, idomain_top_active.data] = mod_id_1d
        return mod_id

    def _render(
        self,
        file: TextIO,
        index: IntArray,
        svat: xr.DataArray,
        mf6_dis: StructuredDiscretization,
        mf6_well: Mf6Wel,
        *args: Any,
    ):
        if mf6_well and (not isinstance(mf6_well, Mf6Wel)):
            raise TypeError(rf"mf6_well not of type 'Mf6Wel', got '{type(mf6_well)}'")
        # Test if equal or larger than 1, to ignore idomain == -1 as well. Don't
        # assign to self.dataset, as grid extent might differ from svat when
        # MetaSWAP only covers part of the Modflow grid domain.
        idomain_active = mf6_dis["idomain"] >= 1
        idomain_top_active = idomain_active.sel(layer=1, drop=True)
        mod_id = self._create_mod_id_rch(svat, idomain_top_active)

        # package check only possible after calling _create_mod_id_rch
        self._pkgcheck()

        data_dict: DataDictType = {"svat": svat.values.ravel()[index]}
        data_dict["layer"] = np.full_like(data_dict["svat"], 1)
        data_dict["mod_id"] = self._index_da(mod_id, index)

        # Get well values
        if mf6_well:
            well_data_dict = self._create_well_id(svat, idomain_active, mf6_well)
            for key in data_dict.keys():
                data_dict[key] = np.append(well_data_dict[key], data_dict[key])

        for var in self._to_fill:
            data_dict[var] = ""

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def _create_well_id(
        self,
        svat: xr.DataArray,
        idomain_active: xr.DataArray,
        mf6_well: Mf6Wel,
    ) -> DataDictType:
        """
        Get modflow indices, svats, and layer number for the wells
        """
        n_subunit = svat["subunit"].size

        well_cellid = mf6_well.dataset["cellid"]
        if len(well_cellid.coords["dim_cellid"]) != 3:
            raise TypeError("Coupling to unstructured grids is not supported.")

        well_layer = well_cellid.sel(dim_cellid="layer").data
        well_row = well_cellid.sel(dim_cellid="row").data - 1
        well_column = well_cellid.sel(dim_cellid="column").data - 1

        n_mod = idomain_active.sum()
        mod_id = xr.full_like(idomain_active, 0, dtype=np.int64)
        mod_id.data[idomain_active.data] = np.arange(1, n_mod + 1)

        well_mod_id = mod_id.data[well_layer - 1, well_row, well_column]
        well_mod_id = np.tile(well_mod_id, (n_subunit, 1))

        well_svat = svat.data[:, well_row, well_column]

        well_active = well_svat != 0

        well_svat_1d = well_svat[well_active]
        well_mod_id_1d = well_mod_id[well_active]

        # Tile well_layers for each subunit
        layer = np.tile(well_layer, (n_subunit, 1))
        layer_1d = layer[well_active]

        return {"mod_id": well_mod_id_1d, "svat": well_svat_1d, "layer": layer_1d}

    def is_regridding_supported(self) -> bool:
        return False
