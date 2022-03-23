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
    svat: array of floats (xr.DataArray)
        SVAT units. This array must have a subunit coordinate to describe
        different land uses.
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

    _with_subunit = ["wel_id", "svat", "layer"]
    _to_fill = ["free"]

    def __init__(self, svat: xr.DataArray, well: mf6.WellDisStructured):
        super().__init__()
        self.well = well
        well_mod_id, well_svat, layer = self._create_well_id(svat)
        self.dataset["wel_id"] = well_mod_id
        self.dataset["svat"] = well_svat
        self.dataset["layer"] = layer

    def _create_well_id(self, svat):
        """
        Get modflow indices, svats, and layer number for the wells
        """
        # Convert to Python's 0-based index
        well_row = self.well["row"] - 1
        well_column = self.well["column"] - 1
        well_layer = self.well["layer"]

        n_subunit, _, _ = svat.shape

        well_svat = svat.values[:, well_row, well_column]
        well_active = well_svat != 0

        well_svat_1d = well_svat[well_active]

        # Tile well_layers for each subunit
        layer = np.tile(well_layer, (n_subunit, 1))
        layer_1d = layer[well_active]

        well_id = self.well.dataset.coords["index"] + 1
        well_id_1d = np.tile(well_id, (n_subunit, 1))[well_active]

        return (well_id_1d, well_svat_1d, layer_1d)

    def _render(self, file, *args):
        data_dict = {}
        data_dict["svat"] = self.dataset["svat"].values
        data_dict["layer"] = self.dataset["layer"].values
        data_dict["wel_id"] = self.dataset["wel_id"].values

        for var in self._to_fill:
            data_dict[var] = ""

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)
