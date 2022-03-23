import numpy as np
import xarray as xr

from imod import mf6
from imod.couplers.metamod.pkgbase import Package
from imod.fixed_format import VariableMetaData


class RechargeSvatMapping(Package):
    """
    This contains the data to map MODFLOW 6 recharge cells to MetaSWAP svats.

    This class is responsible for the file `rchindex2svat.dxc`.

    Unlike imod.msw.CouplerMapping, this class does not include mapping to
    wells.

    Parameters
    ----------
    svat: array of floats (xr.DataArray)
        SVAT units. This array must have a subunit coordinate to describe
        different land uses.
    recharge: mf6.Recharge
        Modflow 6 Recharge package to map to. Note that the recharge rate should
        be provided as a 2D grid with a (y, x) dimension. Package will throw an
        error if a grid is provided with different dimensions.
    """

    # TODO: Do we always want to couple to identical grids?

    _file_name = "rchindex2svat.dxc"
    _metadata_dict = {
        "rch_id": VariableMetaData(10, 1, 9999999, int),
        "free": VariableMetaData(2, None, None, str),
        "svat": VariableMetaData(10, 1, 9999999, int),
        "layer": VariableMetaData(5, 0, 9999, int),
    }

    _with_subunit = ["rch_id", "svat", "layer"]
    _to_fill = ["free"]

    def __init__(self, svat: xr.DataArray, recharge: mf6.Recharge):
        super().__init__()
        self.dataset["svat"] = svat
        self.dataset["layer"] = xr.full_like(svat, 1)
        if "layer" in recharge.dataset.coords:
            self.dataset["rch_active"] = (
                recharge.dataset["rate"].drop("layer").notnull()
            )
        else:
            self.dataset["rch_active"] = recharge.dataset["rate"].notnull()
        self._pkgcheck()
        self._create_rch_id()

    def _create_rch_id(self):
        self.dataset["rch_id"] = xr.full_like(
            self.dataset["svat"], fill_value=0, dtype=np.int64
        )

        n_subunit, _, _ = self.dataset["rch_id"].shape

        subunit = self.dataset.coords["subunit"]
        n_rch = self.dataset["rch_active"].sum()
        valid = self.dataset["rch_active"].expand_dims(subunit=subunit)

        # recharge does not have a subunit dimension, so tile for n_subunits
        rch_id = np.tile(np.arange(1, n_rch + 1), n_subunit)

        self.dataset["rch_id"].values[valid.values] = rch_id

    def _pkgcheck(self):
        rch_dims = self.dataset["rch_active"].dims
        if rch_dims != ("y", "x"):
            raise ValueError(
                f"""Recharge grid can only have dimensions ('y', 'x'). Got
                 {rch_dims} instead"""
            )

        # Check if active msw cell inactive in recharge
        active = self.dataset["svat"] != 0
        inactive_in_rch = active > self.dataset["rch_active"]

        if inactive_in_rch.any():
            raise ValueError(
                """Active MetaSWAP cell detected in inactive cell in Modflow6
                recharge"""
            )
