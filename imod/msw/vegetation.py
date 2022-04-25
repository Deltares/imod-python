import numpy as np
import xarray as xr

from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage


class AnnualCropFactors(MetaSwapPackage):
    """
    For each vegetation type specify a yearly trend in vegetation factors and
    interception characteristics. These are used if WOFOST is not used.

    This class is responsible for the file `fact_svat.inp`.

    Parameters
    ----------
    soil_cover: array of floats (xr.DataArray)
        Soil cover in m2/m2. Must have a "vegetation_index" and "day_of_year" a
        coordinates.
    leaf_area_index: array of floats (xr.DataArray)
        Leaf area index in m2/m2. Must have a "vegetation_index" and
        "day_of_year" a coordinates.
    interception_capacity: array of floats (xr.DataArray)
        Interception capacity in m3/m2. Must have a "vegetation_index" and
        "day_of_year" a coordinates.
    vegetation_factor: array of floats (xr.DataArray)
        Vegetation factor. Must have a "vegetation_index" and "day_of_year" a
        coordinates.
    interception_factor: array of floats (xr.DataArray)
        Interception evaporation factor. Must have a "vegetation_index" and
        "day_of_year" a coordinates.
    bare_soil_factor: array of floats (xr.DataArray)
        Bare soil evaporation factor. Must have a "vegetation_index" and
        "day_of_year" a coordinates.
    ponding_factor: array of floats (xr.DataArray)
        Ponding factor. Must have a "vegetation_index" and "day_of_year" a
        coordinates.
    """

    _file_name = "fact_svat.inp"
    _metadata_dict = {
        "vegetation_index": VariableMetaData(6, 0, 999, int),
        "day_of_year": VariableMetaData(6, 1, 366, int),
        "soil_cover": VariableMetaData(8, 0.0, 1.0, float),
        "leaf_area_index": VariableMetaData(8, 0.0, 10.0, float),
        "interception_capacity": VariableMetaData(8, 0.0, 0.1, float),
        # io manual: min value vegetation_factor = 0.1, but example file has 0.
        # and works
        "vegetation_factor": VariableMetaData(8, 0.0, 10.0, float),
        "interception_factor": VariableMetaData(8, 0.01, 10.0, float),
        "bare_soil_factor": VariableMetaData(8, 0.01, 10.0, float),
        "ponding_factor": VariableMetaData(8, 0.01, 10.0, float),
    }

    def __init__(
        self,
        soil_cover: xr.DataArray,
        leaf_area_index: xr.DataArray,
        interception_capacity: xr.DataArray,
        vegetation_factor: xr.DataArray,
        interception_factor: xr.DataArray,
        bare_soil_factor: xr.DataArray,
        ponding_factor: xr.DataArray,
    ):
        super().__init__()
        self.dataset["soil_cover"] = soil_cover
        self.dataset["leaf_area_index"] = leaf_area_index
        self.dataset["interception_capacity"] = interception_capacity
        self.dataset["vegetation_factor"] = vegetation_factor
        self.dataset["interception_factor"] = interception_factor
        self.dataset["bare_soil_factor"] = bare_soil_factor
        self.dataset["ponding_factor"] = ponding_factor

        self._pkgcheck()

    def _render(self, file, *args):
        dataframe = self.dataset.to_dataframe(
            dim_order=("vegetation_index", "day_of_year")
        ).reset_index()

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def _pkgcheck(self):
        dims = self.dataset.dims
        dims_expected = ("day_of_year", "vegetation_index")
        if len(set(dims) - set(dims_expected)) > 0:
            raise ValueError(
                f"Please provide DataArrays with dimensions {dims_expected}"
            )

        day_of_year = self.dataset.coords["day_of_year"].values
        if not np.all(day_of_year == np.arange(1, 367)):
            raise ValueError(r"Not all days of the year included in data.")
