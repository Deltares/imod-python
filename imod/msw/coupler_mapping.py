import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod.fixed_format import VariableMetaData
from imod.mf6.wel import WellDisStructured
from imod.msw.pkgbase import Package


class CouplerMapping(Package):
    """
    This contains the data to map MODFLOW 6 cells to MetaSWAP svats.

    This class is responsible for the file `mod2svat.inp`.

    Unlike imod.metamod.NodeSvatMapping, this class also includes mapping to wells.

    Parameters
    ----------
    area: array of floats (xr.DataArray)
        Describes the area of SVAT units. This array must have a subunit coordinate
        to describe different land uses.
    active: array of bools (xr.DataArray)
        Describes whether SVAT units are active or not.
        This array must not have a subunit coordinate.
    well: WellDisStructured (optional)
        If given, this parameter describes sprinkling of SVAT units from MODFLOW cells.
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
        well: WellDisStructured = None,
    ):
        super().__init__()
        self.dataset["area"] = area
        self.dataset["active"] = active
        self._create_mod_id_rch()
        self._create_svat()
        self.well = well

    def _create_svat(self):
        self.dataset["svat"] = xr.full_like(
            self.dataset["area"], fill_value=0, dtype=np.int64
        )

        valid = self.dataset["area"].notnull() & self.dataset["active"]
        n_svat = valid.sum()
        self.dataset["svat"].values[valid.values] = np.arange(1, n_svat + 1)

    def _create_mod_id_rch(self):
        """
        Create modflow indices for the recharge layer, which is where
        infiltration will take place.
        """
        _, y_len, x_len = self.dataset["area"].shape
        subunit = self.dataset.coords["subunit"]
        size = self.dataset["active"].size

        mod_id_rch = xr.full_like(self.dataset["active"], fill_value=0, dtype=np.int64)
        mod_id_rch.values = np.arange(1, size + 1).reshape(y_len, x_len)

        self.dataset["mod_id_rch"] = mod_id_rch.expand_dims(subunit=subunit)

    def _render(self, file):
        # Produce values necessary for members without subunit coordinate
        mask = self.dataset["area"].where(self.dataset["active"]).notnull()

        # Generate columns and apply mask
        mod_id = self._get_preprocessed_array("mod_id_rch", mask)
        svat = self._get_preprocessed_array("svat", mask)
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
        """
        Get modflow indices, svats, and layer number for the wells
        """
        # Convert to Python's 0-based index
        well_row = self.well["row"] - 1
        well_column = self.well["column"] - 1
        well_layer = self.well["layer"] - 1

        _, row_len, column_len = self.dataset["svat"].shape
        subunit = self.dataset.coords["subunit"]

        # Check where wells should be coupled to MetaSWAP
        area_valid = ~np.isnan(self.dataset["area"][:, well_row, well_column])
        msw_active = self.dataset["active"][well_row, well_column]
        well_active = area_valid & msw_active

        # Generate modflow indices for cells where wells are located.
        mod_id = (
            well_layer * column_len * row_len + well_row * column_len + well_column + 1
        )
        mod_id = mod_id.expand_dims(subunit=subunit)
        mod_id_1d = mod_id.values[well_active.values]

        # Generate svats
        svat = self.dataset["svat"][:, well_row, well_column]
        svat_1d = svat.values[well_active.values]

        # Generate layers
        layer = well_layer.expand_dims(subunit=subunit)
        # Convert to Modflow's 1-based index. layer is readonly for some reason,
        # so we cannot do += on it.
        layer = layer + 1
        layer_1d = layer.values[well_active.values]

        return (mod_id_1d, svat_1d, layer_1d)

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)
