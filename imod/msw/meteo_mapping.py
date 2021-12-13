import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod.msw.pkgbase import Package, VariableMetaData
from imod.prepare import common


class PrecipitationMapping(Package):
    """
    This contains the data to map precipitation grid cells to MetaSwap svats.

    This class is responsible for the file `svat2precgrid.inp`.
    """

    _file_name = "svat2precgrid.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, None, None, int),
        "row": VariableMetaData(10, None, None, int),
        "column": VariableMetaData(10, None, None, int),
    }

    def __init__(
        self,
        area: xr.DataArray,
        active: xr.DataArray,
        precipitation: xr.DataArray,
    ):
        super().__init__()
        self.dataset["area"] = area
        self.dataset["active"] = active
        self.dataset["precipitation"] = precipitation

    def _render(self, file):
        # Produce values necessary for members without subunit coordinate
        mask = self.dataset["area"].where(self.dataset["active"]).notnull()

        # Generate columns and apply mask
        mod_id = self._get_preprocessed_array("mod_id_rch", mask)
        svat = np.arange(1, mod_id.size + 1)
        layer = np.full_like(svat, 1)

        # Get well values
        if self.well:
            mod_id_well, svat_well, layer_well = self._get_well_values()
            mod_id = np.append(mod_id_well, mod_id)
            svat = np.append(svat_well, svat)
            layer = np.append(layer_well, layer)

        # Generate remaining columns
        free = pd.Series(["" for _ in range(mod_id.size)], dtype="string")

        # Create DataFrame
        dataframe = pd.DataFrame(
            {
                "mod_id": mod_id,
                "free": free,
                "svat": svat,
                "layer": layer,
            }
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def _get_well_values(self):
        mod_id_array = np.array([])
        svat_array = np.array([])
        layer_array = np.array([])

        well_row = self.well["row"]
        well_column = self.well["column"]
        well_layer = self.well["layer"]

        subunit_len, row_len, column_len = self.dataset["svat"].shape

        for row, column, layer in zip(well_row, well_column, well_layer):
            # Convert from 1-indexing to 0 indexing
            row -= 1
            column -= 1
            layer -= 1
            for subunit in range(subunit_len):
                if self.dataset["active"][row, column] and not np.isnan(
                    self.dataset["area"][subunit, row, column]
                ):
                    mod_id = (
                        layer * column_len * row_len + row * column_len + column + 1
                    )
                    mod_id_array = np.append(mod_id_array, mod_id)
                    svat_array = np.append(
                        svat_array, self.dataset["svat"][subunit, row, column]
                    )
                    layer_array = np.append(layer_array, layer + 1)

        return (mod_id_array, svat_array, layer_array)

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)


def grid_mapping(svat_grid: xr.DataArray, meteo_grid: xr.DataArray) -> pd.DataFrame:
    flip_svat_x = svat_grid.indexes["x"].is_monotonic_decreasing
    flip_svat_y = svat_grid.indexes["y"].is_monotonic_decreasing
    flip_meteo_x = meteo_grid.indexes["x"].is_monotonic_decreasing
    flip_meteo_y = meteo_grid.indexes["y"].is_monotonic_decreasing
    nrow = meteo_grid["y"].size
    ncol = meteo_grid["y"].size

    # Ensure all are increasing
    pass

    # Convert to cell boundaries for the meteo grid
    meteo_x = common._coord(meteo_grid, "x")
    meteo_y = common._coord(meteo_grid, "y")

    # Maybe side="left" or side="right" is appropriate...
    index_x = np.searchsorted(svat_grid["x"].values, meteo_x)
    index_y = np.searchsorted(svat_grid["y"].values, meteo_x)

    # Find out of bounds members
    if (index_x == 0).any() or (index_x >= ncol).any():
        raise ValueError("out of bounds for x")
    if (index_y == 0).any() or (index_y >= nrow).any():
        raise ValueError("out of bounds for y")
    rows, columns = np.meshgrid(index_y, index_x, indexing="ij")
    if flip_meteo_x ^ flip_svat_x:  # or something?
        rows = (nrow - 1) - rows

    # Repeat for multiple stacked SVATs I guess?
    return pd.DataFrame({"svat": svat, "row": rows, "column": columns})
