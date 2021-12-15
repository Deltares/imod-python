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
        self.precipitation = precipitation

    def _render(self, file):
        svat_grid = self.dataset["area"].where(self.dataset["active"])

        # Create DataFrame
        dataframe = grid_mapping(svat_grid, self.precipitation)

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
    ncol = meteo_grid["x"].size

    # Convert to cell boundaries for the meteo grid
    meteo_x = common._coord(meteo_grid, "x")
    meteo_y = common._coord(meteo_grid, "y")

    # Create the SVAT grid
    svat_grid_x = np.array([])
    svat_grid_y = np.array([])
    for subunits in svat_grid:
        for rows in subunits:
            for value in rows:
                if not np.isnan(value.values):
                    svat_grid_x = np.append(svat_grid_x, value["x"].values)
                    svat_grid_y = np.append(svat_grid_y, value["y"].values)

    # Determine where the svats fit in within the cell boundaries of the meteo grid
    row = np.searchsorted(meteo_y, svat_grid_y)
    column = np.searchsorted(meteo_x, svat_grid_x)

    # Find out of bounds members
    if (column == 0).any() or (column > ncol).any():
        raise ValueError("Some values are out of bounds for column")
    if (row == 0).any() or (row > nrow).any():
        raise ValueError("Some values are out of bounds for row")

    # Flip axis if necessary
    if flip_meteo_y ^ flip_svat_y:
        row = (nrow + 1) - row
    if flip_meteo_x ^ flip_svat_x:
        column = (column + 1) - column

    # Create svat column
    svat = np.arange(1, svat_grid_x.size + 1)

    return pd.DataFrame({"svat": svat, "row": row, "column": column})
