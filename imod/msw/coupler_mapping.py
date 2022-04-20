import numpy as np
import pandas as pd
import xarray as xr

from imod import mf6
from imod.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage


class CouplerMapping(MetaSwapPackage):
    """
    This contains the data to connect MODFLOW 6 cells to MetaSWAP svats.

    This class is responsible for the file `mod2svat.inp`. It also includes
    connection to wells.

    Parameters
    ----------
    modflow_dis: StructuredDiscretization
        Modflow 6 structured discretization
    well: WellDisStructured (optional)
        If given, this parameter describes sprinkling of SVAT units from MODFLOW
        cells.
    """

    _file_name = "mod2svat.inp"
    _metadata_dict = {
        "mod_id": VariableMetaData(10, 1, 9999999, int),
        "free": VariableMetaData(2, None, None, str),
        "svat": VariableMetaData(10, 1, 9999999, int),
        "layer": VariableMetaData(5, 0, 9999, int),
    }

    _with_subunit = ("mod_id",)
    _without_subunit = ()
    _to_fill = ("free",)

    def __init__(
        self,
        modflow_dis: mf6.StructuredDiscretization,
        well: mf6.WellDisStructured = None,
    ):
        super().__init__()

        self.well = well
        # Test if equal to 1, to ignore idomain == -1 as well.
        # Don't assign to self.dataset, as grid extent might
        # differ from svat
        self.idomain_active = modflow_dis["idomain"] == 1.0

    def _create_mod_id_rch(self, svat):
        """
        Create modflow indices for the recharge layer, which is where
        infiltration will take place.
        """
        self.dataset["mod_id"] = xr.full_like(svat, fill_value=0, dtype=np.int64)
        n_subunit = svat["subunit"].size
        idomain_top_active = self.idomain_active.sel(layer=1, drop=True)

        n_mod_top = idomain_top_active.sum()

        # idomain does not have a subunit dimension, so tile for n_subunits
        mod_id_1d = np.tile(np.arange(1, n_mod_top + 1), (n_subunit, 1))

        self.dataset["mod_id"].values[:, idomain_top_active.values] = mod_id_1d

    def _render(self, file, index, svat):
        self._create_mod_id_rch(svat)
        # package check only possible after calling _create_mod_id_rch
        self._pkgcheck()

        data_dict = {"svat": svat.values.ravel()[index]}

        data_dict["layer"] = np.full_like(data_dict["svat"], 1)

        for var in self._with_subunit:
            data_dict[var] = self._index_da(self.dataset[var], index)

        # Get well values
        if self.well:
            mod_id_well, svat_well, layer_well = self._create_well_id(svat)
            data_dict["mod_id"] = np.append(mod_id_well, data_dict["mod_id"])
            data_dict["svat"] = np.append(svat_well, data_dict["svat"])
            data_dict["layer"] = np.append(layer_well, data_dict["layer"])

        for var in self._to_fill:
            data_dict[var] = ""

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def _create_well_id(self, svat):
        """
        Get modflow indices, svats, and layer number for the wells
        """
        n_subunit = svat["subunit"].size

        # Convert to Python's 0-based index
        well_row = self.well["row"] - 1
        well_column = self.well["column"] - 1
        well_layer = self.well["layer"] - 1

        n_mod = self.idomain_active.sum()
        mod_id = xr.full_like(self.idomain_active, 0, dtype=np.int64)
        mod_id.values[self.idomain_active.values] = np.arange(1, n_mod + 1)

        well_mod_id = mod_id[well_layer, well_row, well_column]
        well_mod_id = np.tile(well_mod_id, (n_subunit, 1))

        well_svat = svat.values[:, well_row, well_column]

        well_active = well_svat != 0

        well_svat_1d = well_svat[well_active]
        well_mod_id_1d = well_mod_id[well_active]

        # Tile well_layers for each subunit
        layer = np.tile(well_layer + 1, (n_subunit, 1))
        layer_1d = layer[well_active]

        return (well_mod_id_1d, well_svat_1d, layer_1d)
