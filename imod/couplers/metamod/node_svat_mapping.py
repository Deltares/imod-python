import numpy as np
import xarray as xr

from imod import mf6
from imod.couplers.metamod.mappingbase import MetaModMapping
from imod.fixed_format import VariableMetaData


class NodeSvatMapping(MetaModMapping):
    """
    This contains the data to connect MODFLOW 6 cells (user nodes) to MetaSWAP
    svats.

    This class is responsible for the file `nodenr2svat.dxc`.

    Parameters
    ----------
    svat: array of floats (xr.DataArray)
        SVAT units. This array must have a subunit coordinate to describe
        different land uses.
    modflow_dis: mf6.StructuredDiscretization
        Modflow 6 structured discretization
    """

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
        self._pkgcheck()
        self._create_mod_id()

    def _create_mod_id(self):
        """
        Create modflow indices for the recharge layer, which is where
        infiltration will take place.
        """
        self.dataset["mod_id"] = xr.full_like(
            self.dataset["svat"], fill_value=0, dtype=np.int64
        )

        n_subunit = self.dataset["subunit"].size
        n_mod = self.idomain_active.sum()

        idomain_active = self.idomain_active.values

        # idomain does not have a subunit dimension, so tile for n_subunits
        mod_id_1d = np.tile(np.arange(1, n_mod + 1), (n_subunit, 1))

        self.dataset["mod_id"].values[:, idomain_active] = mod_id_1d

    def _pkgcheck(self):
        # Check if active msw cell inactive in recharge
        active = self.dataset["svat"] != 0
        inactive_in_idomain = active > self.idomain_active

        if inactive_in_idomain.any():
            raise ValueError(
                "Active MetaSWAP cell detected in inactive cell in Modflow6 idomain"
            )
