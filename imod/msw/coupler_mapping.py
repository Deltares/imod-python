import numpy as np
import pandas as pd
import xarray as xr

from imod.fixed_format import VariableMetaData
from imod.mf6.wel import WellDisStructured
from imod.mf6.dis import StructuredDiscretization
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

    _with_subunit = ["mod_id"]
    _without_subunit = []
    _to_fill = ["free"]

    def __init__(
        self,
        idomain: StructuredDiscretization,
        well: WellDisStructured = None,
    ):
        super().__init__()

        self.well = well
        idomain_top_layer = idomain.sel(layer=1, drop=True)
        # Test if equal to 1, to ignore idomain == -1 as well.
        # Don't assign to self.dataset, as grid extent might
        # differ from svat
        self.idomain_active = idomain_top_layer == 1.0

    def _create_mod_id_rch(self, svat):
        """
        Create modflow indices for the recharge layer, which is where
        infiltration will take place.
        """
        # self.dataset["mod_id"] = xr.full_like(svat, fill_value=0, dtype=np.int64)
        # n_subunit, _, _ = svat.shape
        subunit = self.dataset.coords["subunit"]

        n_mod = self.idomain_active.sum()
        mod_id = xr.full_like(self.idomain_active, fill_value=0, dtype=np.int64)

        ## idomain does not have a subunit dimension, so tile for n_subunits
        # mod_id_1d = np.tile(np.arange(1, n_mod + 1), n_subunit)
        mod_id_1d = np.arange(1, n_mod + 1)

        mod_id.values[self.idomain_active.values] = mod_id_1d

        self.dataset["mod_id"] = mod_id.reindex_like(svat).expand_dims(subunit=subunit)

    def _render(self, file, index, svat):
        self._create_mod_id_rch(svat)

        data_dict = {"svat": svat.values.ravel()[index]}

        data_dict["layer"] = np.full_like(data_dict["svat"], 1)

        for var in self._with_subunit:
            data_dict[var] = self._index_da(self.dataset[var], index)

        # Get well values
        if self.well:
            mod_id_well, svat_well, layer_well = self._get_well_values(svat)
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

    def _get_well_values(self, svat):
        """
        Get modflow indices, svats, and layer number for the wells
        """
        # Convert to Python's 0-based index
        well_row = self.well["row"] - 1
        well_column = self.well["column"] - 1
        well_layer = self.well["layer"] - 1

        modflow_y = self.idomain_active.y[well_row.values]
        modflow_x = self.idomain_active.x[well_column.values]

        y_sel = xr.DataArray(data=modflow_y.values, dims=("index",))
        x_sel = xr.DataArray(data=modflow_x.values, dims=("index",))

        subunit = svat.coords["subunit"]

        well_svat = svat.sel(y=y_sel, x=x_sel).values.ravel()
        well_mod_id = self.dataset["mod_id"].sel(y=y_sel, x=x_sel).values.ravel()

        # alternatively:
        # well_svat = svat[:, well_row, well_column]
        # well_mod_id = self.dataset["mod_id"][:, well_row, well_column]

        well_active = well_svat != 0

        well_svat_1d = well_svat.values[well_active.values]
        well_mod_id_1d = well_mod_id.values[well_active.values]

        layer = well_layer.expand_dims(subunit=subunit)
        # Convert to Modflow's 1-based index. layer is readonly for some reason,
        # so we cannot do += on it.
        layer = layer + 1
        layer_1d = layer.values[well_active.values]

        return (well_mod_id_1d, well_svat_1d, layer_1d)
