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
        self.well = well

    def _create_mod_id_rch(self):
        self.dataset["mod_id_rch"] = self.dataset["area"].copy()
        subunit_len, y_len, x_len = self.dataset["mod_id"].shape
        for subunit in range(subunit_len):
            for y in range(y_len):
                for x in range(x_len):
                    self.dataset["mod_id_rch"][subunit, y, x] = x + y * x_len + 1

    def _render(self, file):
        # Generate columns for members with subunit coordinate
        area = self._get_preprocessed_array("area", self.dataset["active"])

        # Produce values necessary for members without subunit coordinate
        mask = self.dataset["area"].where(self.dataset["active"]).notnull()

        # Generate columns and apply mask
        mod_id_rch = self._get_preprocessed_array("mod_id_rch", mask)

        # Get well values
        if self.well:
            self._get_well_values()

        # Generate remaining columns
        svat = np.arange(1, area.size + 1)
        layer = np.full_like(svat, 1)
        free = pd.Series(["" for _ in range(area.size)], dtype="string")

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
        id_array = []
        layer_array = []
        svat_array = []

        well_row = self.well["row"][1]
        well_column = self.well["column"][1]
        well_layer = self.well["layer"][1]

        subunit_len = self.dataset["area"].shape[0]

        for row, column, layer in zip(well_row, well_column, well_layer):
            for subunit in range(subunit_len):
                if self.dataset["active"][row, column] and not np.isnan(
                    self.dataset["area"][subunit, row, column]
                ):
                    id_array += self.dataset["mod_id_rch"][subunit, row, column]
                    layer_array += layer
                    # TODO: fill svat_array

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)
