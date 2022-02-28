import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod.couplers.metamod.pkgbase import Package
from imod import mf6
from imod.fixed_format import VariableMetaData


class RechargeSvatMapping(Package):
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
    recharge: mf6.Recharge
        Modflow 6 Recharge package to map to.
    """

    # TODO: Do we always want to couple to identical grids?

    _file_name = "rchindex2svat.dxc"
    _metadata_dict = {
        "rch_id": VariableMetaData(10, 1, 9999999, int),
        "free": VariableMetaData(2, None, None, str),
        "svat": VariableMetaData(10, 1, 9999999, int),
        "layer": VariableMetaData(5, 0, 9999, int),
    }

    def __init__(
        self, area: xr.DataArray, active: xr.DataArray, recharge: mf6.Recharge
    ):
        super().__init__()
        self.dataset["area"] = area
        self.dataset["active"] = active
        self.dataset["rch_active"] = recharge.dataset["rate"].notnull()
        self._pkgcheck()
        self._create_rch_id()

    def _create_rch_id(self):
        self.dataset["rch_id"] = self.dataset["area"].copy()

        subunit_len, y_len, x_len = self.dataset["rch_id"].shape

        # Call cumsum on
        rch_id = np.cumsum(self.dataset["rch_active"].values).reshape(y_len, x_len)

        # Copy grid along subunit dimension with broadcasting
        # this is 500 times faster than calling np.stack([rch_id] * subunit_len, axis=0)
        self.dataset["rch_id"].values = np.broadcast_to(
            rch_id, (subunit_len, y_len, x_len)
        )

        self.dataset["rch_id"] = self.dataset["rch_id"].where(
            self.dataset["rch_active"]
        )

    def _render(self, file):
        # Produce values necessary for members with subunit coordinate
        mask = self.dataset["area"].where(self.dataset["active"]).notnull()

        # Generate columns and apply mask
        rch_id = self._get_preprocessed_array("rch_id", mask)

        # Generate remaining columns
        # TODO: In GridData, svat is generated as np.arange(1, area.size + 1)
        svat = np.arange(1, rch_id.size + 1)
        # TODO: Always stuck to layer 1? At least add to docstring!
        layer = np.full_like(svat, 1)
        free = pd.Series(["" for _ in range(rch_id.size)], dtype="string")

        # Create DataFrame
        dataframe = pd.DataFrame(
            {
                "rch_id": rch_id,
                "free": free,
                "svat": svat,
                "layer": layer,
            }
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def _pkgcheck(self):
        # Check if active msw cell inactive in recharge
        inactive_in_rch = self.dataset["active"] > self.dataset["rch_active"]

        if inactive_in_rch.any():
            raise ValueError(
                "Active MetaSWAP cell detected in inactive cell in Modflow6 recharge"
            )

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)
