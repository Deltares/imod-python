import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod import mf6
from imod.couplers.metamod.pkgbase import Package
from imod.fixed_format import VariableMetaData


class WellSvatMapping(Package):
    """
    This contains the data to map MODFLOW 6 recharge cells to MetaSWAP svats.

    This class is responsible for the file `rchindex2svat.dxc`.

    Unlike imod.msw.CouplerMapping, this class does not include mapping to wells.

    Parameters
    ----------
    area: array of floats (xr.DataArray)
        Describes the area of SVAT units. This array must have a subunit coordinate
        to describe different land uses.
    active: array of bools (xr.DataArray)
        Describes whether SVAT units are active or not.
        This array must not have a subunit coordinate.
    well: mf6.Well
        Modflow 6 Well package to map to.
    """

    # TODO: Do we always want to couple to identical grids?

    _file_name = "wellindex2svat.dxc"
    _metadata_dict = {
        "wel_id": VariableMetaData(10, 1, 9999999, int),
        "free": VariableMetaData(2, None, None, str),
        "svat": VariableMetaData(10, 1, 9999999, int),
        "layer": VariableMetaData(5, 0, 9999, int),
    }

    def __init__(
        self, area: xr.DataArray, active: xr.DataArray, well: mf6.WellDisStructured
    ):
        super().__init__()
        self.dataset["area"] = area
        self.dataset["active"] = active
        self.well = well
        self._create_svat()
        self._create_wel_id()

    def _create_svat(self):
        self.dataset["svat"] = xr.full_like(
            self.dataset["area"], fill_value=0, dtype=np.int64
        )

        valid = self.dataset["area"].notnull() & self.dataset["active"]
        n_svat = valid.sum()
        self.dataset["svat"].values[valid.values] = np.arange(1, n_svat + 1)

    def _create_wel_id(self):
        # Convert to Python's 0-based index
        well_row = self.well["row"] - 1
        well_column = self.well["column"] - 1
        well_layer = self.well["layer"] - 1

        subunit = self.dataset.coords["subunit"]

        # Check where wells should be coupled to MetaSWAP
        area_valid = ~np.isnan(self.dataset["area"][:, well_row, well_column])
        msw_active = self.dataset["active"][well_row, well_column]
        well_active = area_valid & msw_active

        # Select wells
        well_id = self.well.dataset.coords["index"] + 1
        well_id = well_id.expand_dims(subunit=subunit)
        well_id_1d = well_id.values[well_active.values]

        # Generate svats
        svat = self.dataset["svat"][:, well_row, well_column]
        svat_1d = svat.values[well_active.values]

        # Generate layers
        layer = well_layer.expand_dims(subunit=subunit)
        # Convert to Modflow's 1-based index. layer is readonly for some reason,
        # so we cannot do += on it.
        layer = layer + 1
        layer_1d = layer.values[well_active.values]

        return well_id_1d, svat_1d, layer_1d

    def _render(self, file):
        wel_id, svat, layer = self._create_wel_id()

        free = pd.Series(["" for _ in range(wel_id.size)], dtype="string")

        # Create DataFrame
        dataframe = pd.DataFrame(
            {
                "wel_id": wel_id,
                "free": free,
                "svat": svat,
                "layer": layer,
            }
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)
