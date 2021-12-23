import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6 import WellDisStructured
from imod.msw.pkgbase import Package
from imod.fixed_format import VariableMetaData


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
    active: array of bools (xr.DataArray)
        Describes whether SVAT units are active or not.
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

    def __init__(
        self,
        max_abstraction_groundwater: xr.DataArray,
        max_abstraction_surfacewater: xr.DataArray,
        active: xr.DataArray,
        well: WellDisStructured,
    ):
        super().__init__()
        self.dataset["max_abstraction_groundwater_m3_d"] = max_abstraction_groundwater
        self.dataset["max_abstraction_surfacewater_m3_d"] = max_abstraction_surfacewater
        self.dataset["active"] = active
        self.well = well

    def _render(self, file):
        # Preprocess input
        max_abstraction_groundwater_m3_d = self._get_preprocessed_array(
            "max_abstraction_groundwater_m3_d", self.dataset["active"]
        )

        max_abstraction_surfacewater_m3_d = self._get_preprocessed_array(
            "max_abstraction_surfacewater_m3_d", self.dataset["active"]
        )

        # Generate remaining columns
        svat = np.arange(1, max_abstraction_groundwater_m3_d.size + 1)
        max_abstraction_groundwater_mm_d = pd.Series(
            ["" for _ in range(max_abstraction_groundwater_m3_d.size)], dtype="string"
        )
        max_abstraction_surfacewater_mm_d = pd.Series(
            ["" for _ in range(max_abstraction_groundwater_m3_d.size)], dtype="string"
        )
        svat_groundwater = pd.Series(
            ["" for _ in range(max_abstraction_groundwater_m3_d.size)], dtype="string"
        )
        trajectory = pd.Series(
            ["" for _ in range(max_abstraction_groundwater_m3_d.size)], dtype="string"
        )

        # Get layer
        layer = self._get_layer()

        # Create DataFrame
        dataframe = pd.DataFrame(
            {
                "svat": svat,
                "max_abstraction_groundwater_mm_d": max_abstraction_groundwater_mm_d,
                "max_abstraction_surfacewater_mm_d": max_abstraction_surfacewater_mm_d,
                "max_abstraction_groundwater_m3_d": max_abstraction_groundwater_m3_d,
                "max_abstraction_surfacewater_m3_d": max_abstraction_surfacewater_m3_d,
                "svat_groundwater": svat_groundwater,
                "layer": layer,
                "trajectory": trajectory,
            }
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)

    def _get_layer(self):
        # Build up well_dict
        well_dict = {}
        well_row = self.well["row"]
        well_column = self.well["column"]
        well_layer = self.well["layer"]
        for row, column, layer in zip(well_row, well_column, well_layer):
            # Convert from 1-indexing to 0 indexing
            key = (int(row) - 1, int(column) - 1)
            if key in well_dict:
                raise ValueError(
                    "A single svat cannot be sprinkled by multiple groundwater cells."
                )

            well_dict[key] = layer

        # Build up layer_array
        layer_list = []
        row_len, column_len = self.dataset["max_abstraction_groundwater_m3_d"].shape

        for column in range(column_len):
            for row in range(row_len):
                if self.dataset["active"][row, column] and not np.isnan(
                    self.dataset["max_abstraction_groundwater_m3_d"][row, column]
                ):
                    layer_list.append(well_dict[(row, column)])

        return np.array(layer_list)
