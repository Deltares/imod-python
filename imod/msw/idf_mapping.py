import numpy as np
import xarray as xr

from imod.fixed_format import VariableMetaData
from imod.msw.pkgbase import Package
from imod.util import spatial_reference


class IdfMapping(Package):
    """
    Mapping for IDFs.

    Note that MetaSWAP can only write equidistant grids.
    """

    _file_name = "idf_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 9999999, int),
        "rows": VariableMetaData(10, 1, 9999999, int),
        "columns": VariableMetaData(10, 1, 9999999, int),
        # TODO: Check if x and y limits are properly set.
        "y_grid": VariableMetaData(15, 0.0, 9999999.0, float),
        "x_grid": VariableMetaData(15, 0.0, 9999999.0, float),
    }

    _with_subunit = []
    _without_subunit = ["rows", "columns", "y_grid", "x_grid"]
    _to_fill = []

    # TODO: Quote from IO manual: The x- and y-coordinates should increase with increasing col, row.
    # But example works with decreasing y-coordinates?

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
        if (type(dx) is np.ndarray) or (type(dy) is np.ndarray):
            raise ValueError("MetaSWAP can only write equidistant IDF grids")

        nodata = self.dataset["nodata"].values

        # TODO: Check if netcdf_per is also required, as manual seems a bit vague
        # about this. "To activate the idf-option the netcdf_per parameter in
        # PARA_SIM.INP must be set to 1." Could be a typo.
        return dict(
            simgro_opt=-1,
            idf_per=1,
            idf_dx=dx,
            idf_dy=np.abs(dy),
            idf_ncol=ncol,
            idf_nrow=nrow,
            idf_xmin=xmin,
            idf_ymin=ymin,
            idf_nodata=nodata,
        )
