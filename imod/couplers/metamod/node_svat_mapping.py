import numpy as np
import xarray as xr

from imod.couplers.metamod.pkgbase import Package
from imod.fixed_format import VariableMetaData
from imod import mf6


class NodeSvatMapping(Package):
    """
    This contains the data to map MODFLOW 6 cells (user nodes) to MetaSWAP
    svats.

    This class is responsible for the file `nodenr2svat.dxc`.

    Unlike imod.msw.CouplerMapping, this class does not include mapping to
    wells.

    Parameters
    ----------
    svat: array of floats (xr.DataArray)
        SVAT units. This array must have a subunit coordinate to describe
        different land uses.
    modflow_dis: mf6.StructuredDiscretization
        Modflow 6 structured discretization
    """

    # TODO: Package checks:
    #   - Make sure "area==np.nan" and "idomain==0" in the same cells.

    _file_name = "nodenr2svat.dxc"
    _metadata_dict = {
        "mod_id": VariableMetaData(10, 1, 9999999, int),
        "free": VariableMetaData(2, None, None, str),
        "svat": VariableMetaData(10, 1, 9999999, int),
        "layer": VariableMetaData(5, 0, 9999, int),
    }

    _with_subunit = ["mod_id", "svat", "layer"]
    _to_fill = ["free"]

    def __init__(
        self,
        svat: xr.DataArray,
        modflow_dis: mf6.StructuredDiscretization,
    ):
        super().__init__()
        self.dataset["svat"] = svat
        self.dataset["layer"] = xr.full_like(svat, 1)
        idomain_top_layer = modflow_dis["idomain"].sel(layer=1, drop=True)
        # Test if equal to 1, to ignore idomain == -1 as well.
        # Don't assign to self.dataset, as grid extent might
        # differ from svat
        self.idomain_active = idomain_top_layer == 1.0
        self._create_mod_id()

    def _create_mod_id(self):
        """
        Create modflow indices for the recharge layer, which is where
        infiltration will take place.
        """
        self.dataset["mod_id"] = xr.full_like(
            self.dataset["svat"], fill_value=0, dtype=np.int64
        )
        n_subunit, _, _ = self.dataset["svat"].shape
        n_mod = self.idomain_active.sum()

        # idomain does not have a subunit dimension, so tile for n_subunits
        mod_id_1d = np.tile(np.arange(1, n_mod + 1), n_subunit)

        self.dataset["mod_id"].values[:, self.idomain_active.values] = mod_id_1d
