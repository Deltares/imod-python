import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.wel import Well
from imod.msw.pkgbase import Package, VariableMetaData


class CouplerMapping(Package):
    """
    This contains the data to map MODFLOW 6 cells to MetaSwap svats.

    This class is responsible for the file `mod2svat.inp`.

    Unlike imod.metamod.NodeSvatMapping, this class also includes mapping to wells.
    """

    _file_name = "mod2svat.inp"
    _metadata_dict = {
        "mod_id": VariableMetaData(10, 1, 9999999, int),
        "free": VariableMetaData(2, None, None, str),
        "svat": VariableMetaData(10, 1, 9999999, int),
        "layer": VariableMetaData(5, 0, 9999, int),
    }

    def __init__(
        self,
        area: xr.DataArray,
        active: xr.DataArray,
        well: Well = None,
    ):
        super().__init__()
        self.dataset["area"] = area
        self.dataset["active"] = active
        self._create_mod_id_rch()
        self._create_svat()
        self.well = well

    def _create_svat(self):
        self.dataset["svat"] = self.dataset["area"].copy()
        subunit_len, y_len, x_len = self.dataset["svat"].shape

        svat_index = 1
        for subunit in range(subunit_len):
            for y in range(y_len):
                for x in range(x_len):
                    if self.dataset["active"][y, x] and not np.isnan(
                        self.dataset["area"][subunit, y, x]
                    ):
                        self.dataset["svat"][subunit, y, x] = svat_index
                        svat_index += 1

    def _create_mod_id_rch(self):
        self.dataset["mod_id_rch"] = self.dataset["area"].copy()
        subunit_len, y_len, x_len = self.dataset["mod_id_rch"].shape
        for subunit in range(subunit_len):
            for y in range(y_len):
                for x in range(x_len):
                    self.dataset["mod_id_rch"][subunit, y, x] = y * x_len + x + 1

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
        mod_id_array = []
        svat_array = []
        layer_array = []

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
                    mod_id_array += mod_id
                    svat_array += self.dataset["svat"][subunit, row, column]

                    layer_array += layer + 1

        return (np.array(mod_id_array), np.array(svat_array), np.array(layer_array))

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)
