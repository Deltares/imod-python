from typing import Any, TextIO

import pandas as pd
import xarray as xr

from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import DataDictType, MetaSwapPackage
from imod.msw.regrid.regrid_schemes import PondingRegridMethod
from imod.msw.utilities.common import concat_imod5
from imod.typing import Imod5DataDict, IntArray


class Ponding(MetaSwapPackage, IRegridPackage):
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

    _regrid_method = PondingRegridMethod()

    def __init__(self, ponding_depth, runon_resistance, runoff_resistance) -> None:
        super().__init__()
        self.dataset["ponding_depth"] = ponding_depth
        self.dataset["runon_resistance"] = runon_resistance
        self.dataset["runoff_resistance"] = runoff_resistance

        self._pkgcheck()

    def _render(self, file: TextIO, index: IntArray, svat: xr.DataArray, *args: Any):
        data_dict: DataDictType = {"svat": svat.values.ravel()[index]}

        for var in self._with_subunit:
            data_dict[var] = self._index_da(self.dataset[var], index)

        data_dict["swnr"] = 0

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict) -> "Ponding":
        """
        Construct a MetaSWAP Ponding package from iMOD5 data in the CAP
        package, loaded with the :func:`imod.formats.prj.open_projectfile_data`
        function.

        Method concatenates ponding depths, runon resistance, and runoff
        resistance along two subunits. Subunit 0 for rural, and 1 for urban
        landuse.

        Parameters
        ----------
        imod5_data: Imod5DataDict
            iMOD5 data as returned by
            :func:`imod.formats.prj.open_projectfile_data`

        Returns
        -------
        imod.msw.Ponding
        """
        cap_data = imod5_data["cap"]
        data = {}
        for key in cls._with_subunit:
            data_ls = [cap_data[f"{landuse}_{key}"] for landuse in ["rural", "urban"]]
            data[key] = concat_imod5(*data_ls)

        return cls(**data)
