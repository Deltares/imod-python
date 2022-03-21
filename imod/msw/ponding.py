import numpy as np
import pandas as pd

from imod.fixed_format import VariableMetaData
from imod.msw.pkgbase import Package


class Ponding(Package):
    """
    Set ponding related parameters for MetaSWAP. This class is responsible for
    the svat2swnr_roff.inp file. Currently, we do not support ponds coupled to
    SIMGRO's surface water module.

    Parameters
    ----------
    ponding_depth: array of floats (xr.DataArray)
        Ponding depth of the SVAT units in meters. If set to 0. water can freely
        flow over the soil surface. Runoff is disable by setting the ponding
        depth to 9999 m. Large values, e.g. 1000 m, should be avoided becauses
        this causes excess memory use. This array must have a subunit coordinate
        to describe different land uses.
    runoff_resistance: array of floats (xr.DataArray)
        Runoff resistance of SVAT units in days. This array must have a subunit
        coordinate to describe different land uses.
    runon_resistance: array of floats (xr.DataArray)
        Runon resistance of SVAT units in days. This array must have a subunit
        coordinate to describe different land uses.
    active: array of booleans (xr.DataArray)
        Describes whether SVAT units are active or not. This array must not have
        a subunit coordinate.
    """

    _file_name = "svat2swnr_roff.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "swnr": VariableMetaData(10, 0, 99999999, int),
        "ponding_depth": VariableMetaData(8, 0.0, 1e6, float),
        "runoff_resistance": VariableMetaData(8, 0.0, 1e6, float),
        "runon_resistance": VariableMetaData(8, 0.0, 1e6, float),
    }

    def __init__(
        self, ponding_depth, runon_resistance, runoff_resistance, active
    ) -> None:
        super().__init__()
        self.dataset["ponding_depth"] = ponding_depth
        self.dataset["runon_resistance"] = runon_resistance
        self.dataset["runoff_resistance"] = runoff_resistance
        self.dataset["active"] = active

    def _render(self, file):
        # Generate columns for members with subunit coordinate
        ponding_depth = self._get_preprocessed_array(
            "ponding_depth", self.dataset["active"]
        )

        runoff_resistance = self._get_preprocessed_array(
            "runoff_resistance", self.dataset["active"]
        )

        runon_resistance = self._get_preprocessed_array(
            "runon_resistance", self.dataset["active"]
        )

        # Generate remaining columns
        svat = np.arange(1, ponding_depth.size + 1)
        swnr = np.zeros(ponding_depth.size)

        # Create DataFrame
        dataframe = pd.DataFrame(
            {
                "svat": svat,
                "swnr": swnr,
                "ponding_depth": ponding_depth,
                "runon_resistance": runon_resistance,
                "runoff_resistance": runoff_resistance,
            }
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)
