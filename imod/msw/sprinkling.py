import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.wel import WellDisStructured
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage


class Sprinkling(MetaSwapPackage):
    """
    This contains the sprinkling capacities of links between SVAT units and
    groundwater/surface water locations.

    This class is responsible for the file `scap_svat.inp`

    Parameters
    ----------
    max_abstraction_groundwater: array of floats (xr.DataArray)
        Describes the maximum abstraction of groundwater to SVAT units in m3 per
        day. This array must not have a subunit coordinate.
    max_abstraction_surfacewater: array of floats (xr.DataArray)
        Describes the maximum abstraction of surfacewater to SVAT units in m3
        per day. This array must not have a subunit coordinate.
    well_id: array of int (xr.DataArray) 
        Describes per svat the corresponing id of the well package
        from which water is extracted.
    well: WellDisStructured
        Describes the source location for groundwater intake.
    """

    _file_name = "scap_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "max_abstraction_groundwater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_surfacewater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_groundwater_m3_d": VariableMetaData(8, 0.0, 1e9, float),
        "max_abstraction_surfacewater_m3_d": VariableMetaData(8, 0.0, 1e9, float),
        "svat_groundwater": VariableMetaData(10, 1, 99999999, int),
        "layer": VariableMetaData(6, 1, 9999, int),
        "trajectory": VariableMetaData(10, None, None, str),
    }

    _with_subunit = (        
        "max_abstraction_groundwater_m3_d",
        "max_abstraction_surfacewater_m3_d",)
    _without_subunit = ()

    _to_fill = (
        "max_abstraction_groundwater_mm_d",
        "max_abstraction_surfacewater_mm_d",
        "trajectory",
    )

    def __init__(
        self,
        max_abstraction_groundwater: xr.DataArray,
        max_abstraction_surfacewater: xr.DataArray,
        well: WellDisStructured,
    ):
        super().__init__()
        self.dataset["max_abstraction_groundwater_m3_d"] = max_abstraction_groundwater
        self.dataset["max_abstraction_surfacewater_m3_d"] = max_abstraction_surfacewater
        self.well = well

        self._pkgcheck()    


    def _render(self, file, index, svat):
        def ravel_per_subunit(array: xr.DataArray) -> np.ndarray:
            # per defined well element, all subunits
            array_out = array.to_numpy()[:, well_row, well_column].ravel()
            # per defined well element, per defined subunits
            return array_out[np.isfinite(array_out)]
        
        well_row = self.well["row"] - 1
        well_column = self.well["column"] - 1
        well_layer = self.well["layer"]
        max_rate_per_svat = self.dataset["max_abstraction_groundwater_m3_d"].where(
            svat > 0
        )
        layer_per_svat = xr.full_like(max_rate_per_svat, np.nan)
        layer_per_svat[:, well_row, well_column] = well_layer

        layer_source = ravel_per_subunit(
            layer_per_svat.where(max_rate_per_svat > 0)
        ).astype(dtype=np.int32)
        svat_source_target = ravel_per_subunit(
            svat.where(max_rate_per_svat > 0)
        ).astype(dtype=np.int32)

        data_dict = {
            "svat": svat_source_target,
            "layer": layer_source,
            "svat_groundwater": svat_source_target,
        }
        

        for var in self._with_subunit:
            array = self.dataset[var].where(max_rate_per_svat > 0).to_numpy()
            array = array[np.isfinite(array)]
            data_dict[var] = array

        for var in self._to_fill:
            data_dict[var] = ""

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )
        
        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)


class SprinklingMultipleSources(Sprinkling):
    
    def __init__(
        self,
        max_abstraction_groundwater: xr.DataArray,
        max_abstraction_surfacewater: xr.DataArray,
        well: WellDisStructured,
        well_id: xr.DataArray
    ):
        super().__init__(max_abstraction_groundwater, max_abstraction_surfacewater, well)
        self.well_id = well_id
        
        
    def _render(self, file, index, svat):
        well_row = self.well["row"] - 1
        well_column = self.well["column"] - 1
        layer_source = self.well["layer"]
        max_rate = self.dataset["max_abstraction_groundwater_m3_d"]
        
        svat_target = svat.where(max_rate > 0).to_numpy()
        svat_target = svat_target[np.isfinite(svat_target)].astype(dtype=np.int32)
        
        well_id = self.well_id.where(max_rate > 0).to_numpy()
        well_id = well_id[np.isfinite(well_id)].astype(dtype=np.int32)
        
        # always use the first svat as source
        svat_source = svat.to_numpy()[0, well_row[well_id], well_column[well_id]]
        layer_source = layer_source[well_id]

        data_dict = {"svat": svat_target, "layer": layer_source, "svat_groundwater": svat_source}

        for var in self._with_subunit:
            array = self.dataset[var].where(max_rate > 0).to_numpy()
            array = array[np.isfinite(array)]
            data_dict[var] = array

        for var in self._to_fill:
            data_dict[var] = ""

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)