import numpy as np
import pandas as pd
import xarray as xr

from imod.fixed_format import VariableMetaData
from imod.msw.pkgbase import Package

from imod import mf6


class Sprinkling(Package):
    """
    This contains the sprinkling capacities of links between SVAT units and groundwater/ surface water locations.

    This class is responsible for the file `scap_svat.inp`

    Parameters
    ----------
    max_abstraction_groundwater: array of floats (xr.DataArray)
        Describes the maximum abstraction of groundwater to SVAT units in m3 per day.
        This array must not have a subunit coordinate.
    max_abstraction_surfacewater: array of floats (xr.DataArray)
        Describes the maximum abstraction of surfacewater to SVAT units in m3 per day.
        This array must not have a subunit coordinate.
    well: WellDisStructured
        Describes the sprinkling of SVAT units coming groundwater.
    """

    _file_name = "scap_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "max_abstraction_groundwater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_surfacewater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_groundwater_m3_d": VariableMetaData(8, 0.0, 1e9, float),
        "max_abstraction_surfacewater_m3_d": VariableMetaData(8, 0.0, 1e9, float),
        "svat_groundwater": VariableMetaData(10, None, None, str),
        "layer": VariableMetaData(6, 1, 9999, int),
        "trajectory": VariableMetaData(10, None, None, str),
    }

    _with_subunit = []
    _without_subunit = [
        "max_abstraction_groundwater_m3_d",
        "max_abstraction_surfacewater_m3_d",
    ]

    _to_fill = [
        "max_abstraction_groundwater_mm_d",
        "max_abstraction_surfacewater_mm_d",
        "svat_groundwater",
        "trajectory",
    ]

    def __init__(
        self,
        max_abstraction_groundwater: xr.DataArray,
        max_abstraction_surfacewater: xr.DataArray,
        well: mf6.WellDisStructured,
    ):
        super().__init__()
        self.dataset["max_abstraction_groundwater_m3_d"] = max_abstraction_groundwater
        self.dataset["max_abstraction_surfacewater_m3_d"] = max_abstraction_surfacewater
        self.well = well

    def _render(self, file, index, svat):
        well_row = self.well["row"] - 1
        well_column = self.well["column"] - 1
        well_layer = self.well["layer"]

        n_subunit, _, _ = svat.shape

        well_svat = svat.values[:, well_row, well_column]
        well_active = well_svat != 0

        # Tile well_layers for each subunit
        layer = np.tile(well_layer, (n_subunit, 1))

        data_dict = {"svat": well_svat[well_active], "layer": layer[well_active]}

        for var in self._without_subunit:
            well_arr = self.dataset[var].values[well_row, well_column]
            well_arr = np.tile(well_arr, (n_subunit, 1))
            data_dict[var] = well_arr[well_active]

        for var in self._to_fill:
            data_dict[var] = ""

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)
