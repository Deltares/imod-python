import pandas as pd

from imod.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage


class Ponding(MetaSwapPackage):
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
    """

    _file_name = "svat2swnr_roff.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "swnr": VariableMetaData(10, 0, 99999999, int),
        "ponding_depth": VariableMetaData(8, 0.0, 1e6, float),
        "runoff_resistance": VariableMetaData(8, 0.0, 1e6, float),
        "runon_resistance": VariableMetaData(8, 0.0, 1e6, float),
    }

    _with_subunit = ("ponding_depth", "runoff_resistance", "runon_resistance")
    _without_subunit = ()
    _to_fill = ()

    def __init__(self, ponding_depth, runon_resistance, runoff_resistance) -> None:
        super().__init__()
        self.dataset["ponding_depth"] = ponding_depth
        self.dataset["runon_resistance"] = runon_resistance
        self.dataset["runoff_resistance"] = runoff_resistance

        self._pkgcheck()

    def _render(self, file, index, svat):
        data_dict = {"svat": svat.values.ravel()[index]}

        for var in self._with_subunit:
            data_dict[var] = self._index_da(self.dataset[var], index)

        data_dict["swnr"] = 0

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)
