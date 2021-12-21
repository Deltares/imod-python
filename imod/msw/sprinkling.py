import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6 import Well
from imod.msw.pkgbase import Package, VariableMetaData


class Sprinkling(Package):
    """
    This contains the sprinkling capacities of links between SVAT units and groundwater/ surface water locations.

    This class is responsible for the file `scap_svat.inp`
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
        well: Well,
        active: xr.DataArray,
    ):
        super().__init__()
        self.dataset["max_abstraction_groundwater"] = max_abstraction_groundwater
        self.dataset["max_abstraction_surfacewater"] = max_abstraction_surfacewater
        self.dataset["active"] = active
        self.well = well

    def _render(self, file):
        # Preprocess input
        max_abstraction_groundwater_m3_d = self._get_preprocessed_array(
            "max_abstraction_groundwater", self.dataset["active"]
        )

        max_abstraction_surfacewater_m3_d = self._get_preprocessed_array(
            "max_abstraction_surfacewater", self.dataset["active"]
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
        layer_array = []

        well_row = self.well["row"]
        well_column = self.well["column"]
        well_layer = self.well["layer"]

        for row, column, layer in zip(well_row, well_column, well_layer):
            # Convert from 1-indexing to 0 indexing
            row -= 1
            column -= 1
            layer -= 1

            if self.dataset["active"][row, column]:
                if np.isnan(self.dataset["max_abstraction_groundwater"][row, column]):
                    raise ValueError(
                        "max_abstraction_groundwater must to be defined for all wells."
                    )

                layer_array.append(
                    self.dataset["max_abstraction_groundwater"][row, column]
                )
        return np.array(layer_array)
