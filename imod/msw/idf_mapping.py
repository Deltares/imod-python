from dataclasses import asdict
from typing import Optional

import numpy as np
import xarray as xr

from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.utilities.regrid import RegridderWeightsCache, _regrid_array
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.regrid.regrid_schemes import IdfMappingRegridMethod
from imod.typing import GridDataArray
from imod.util.regrid_method_type import RegridMethodType
from imod.util.spatial import spatial_reference


class IdfMapping(MetaSwapPackage, IRegridPackage):
    """
    Describes svat location in the IDF grid.

    Note that MetaSWAP can only write equidistant grids.
    """

    _file_name = "idf_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 9999999, int),
        "rows": VariableMetaData(10, 1, 9999999, int),
        "columns": VariableMetaData(10, 1, 9999999, int),
        "y_grid": VariableMetaData(15, -9999999.0, 9999999.0, float),
        "x_grid": VariableMetaData(15, -9999999.0, 9999999.0, float),
    }

    _with_subunit = ()
    _without_subunit = ("rows", "columns", "y_grid", "x_grid")
    _to_fill = ()

    _regrid_method = IdfMappingRegridMethod()

    # NOTE that it is stated in the IO manual: "The x- and y-coordinates should
    # increase with increasing col, row." But the example works with decreasing
    # y-coordinates.

    def __init__(self, area, nodata):
        super().__init__()

        self.dataset["area"] = area
        self.dataset["nodata"] = nodata

        nrow = self.dataset.coords["y"].size
        ncol = self.dataset.coords["x"].size

        y_index = xr.DataArray(
            np.arange(1, nrow + 1), coords={"y": self.dataset.coords["y"]}, dims=("y",)
        )
        x_index = xr.DataArray(
            np.arange(1, ncol + 1), coords={"x": self.dataset.coords["x"]}, dims=("x",)
        )
        rows, columns = xr.broadcast(y_index, x_index)

        self.dataset["rows"] = rows
        self.dataset["columns"] = columns

        y_grid, x_grid = xr.broadcast(self.dataset["y"], self.dataset["x"])

        self.dataset["x_grid"] = x_grid
        self.dataset["y_grid"] = y_grid

    def get_output_settings(self):
        grid = self.dataset["area"]
        dx, xmin, _, dy, ymin, _ = spatial_reference(grid)
        ncol = grid["x"].size
        nrow = grid["y"].size

        # If non-equidistant, spatial_reference returned a 1d array instead of
        # float
        if (not np.isscalar(dx)) or (not np.isscalar(dy)):
            raise ValueError("MetaSWAP only supports equidistant grids")

        nodata = self.dataset["nodata"].values

        return {
            "simgro_opt": -1,
            "idf_per": 1,
            "idf_dx": dx,
            "idf_dy": np.abs(dy),
            "idf_ncol": ncol,
            "idf_nrow": nrow,
            "idf_xmin": xmin,
            "idf_ymin": ymin,
            "idf_nodata": nodata,
        }

    def regrid_like(
        self,
        target_grid: GridDataArray,
        regrid_context: RegridderWeightsCache,
        regridder_types: Optional[RegridMethodType] = None,
    ) -> "MetaSwapPackage":
        if regridder_types is None:
            regridder_settings = asdict(self.get_regrid_methods(), dict_factory=dict)
        else:
            regridder_settings = asdict(regridder_types, dict_factory=dict)

        nodata = self.dataset["nodata"].values[()]
        regridded_area = _regrid_array(
            self.dataset["area"],
            regrid_context,
            regridder_settings["area"][0],
            regridder_settings["area"][1],
            target_grid,
        )

        return type(self)(regridded_area, nodata)
