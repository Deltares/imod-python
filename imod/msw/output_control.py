from imod.msw.pkgbase import Package
from imod.fixed_format import VariableMetaData
from imod.util import spatial_reference
import numpy as np
import xarray as xr
import pathlib
import pandas as pd


class OutputControl(Package):
    # TODO: Get list from Joachim which files we want to support, as there
    # are a lot options, many of which we never use.

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_settings(self):
        """
        Return relevant settings for the PARA_SIM.INP file
        """
        return self._settings


class IdfOutputControl(OutputControl):
    """
    Output control to generate IDFs.

    Note that MetaSWAP can only write equidistant grids.
    """

    _file_name = "idf_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 9999999, int),
        "rows": VariableMetaData(10, 1, 9999999, int),
        "columns": VariableMetaData(10, 1, 9999999, int),
        # TODO: Check if x and y limits are properly set.
        "y_coords": VariableMetaData(12, 0.0, 9999999.0, float),
        "x_coords": VariableMetaData(12, 0.0, 9999999.0, float),
    }

    # TODO: Quote from IO manual: The x- and y-coordinates should increase with increasing col, row.
    # But example works with decreasing y-coordinates?

    def __init__(self, area, active, nodata):
        super().__init__()

        self.dataset["area"] = area
        self.dataset["active"] = active
        self.dataset["nodata"] = nodata

        nrow = self.dataset.coords["y"].size
        ncol = self.dataset.coords["x"].size

        # TODO: Refactor _get_preprocessed_array to accept a DataArray as
        # argument insteat of a varname. Then this assigning becomes
        # unnecessary.
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

    def get_settings(self):
        grid = self.dataset["area"]
        dx, xmin, _, dy, ymin, _ = spatial_reference(grid)
        ncol = grid["x"].size
        nrow = grid["y"].size

        # If non-equidistant, spatial_reference returned a 1d array instead of
        # float
        if (type(dx) is np.ndarray) or (type(dy) is np.ndarray):
            raise ValueError("MetaSWAP can only write equidistant IDF grids")

        nodata = self.dataset["nodata"].values

        return dict(
            simgro_opt=-1,
            idf_per=1,
            idf_dx=dx,
            idf_dy=dy,
            idf_ncol=ncol,
            idf_nrow=nrow,
            idf_xmin=xmin,
            idf_ymin=ymin,
            idf_nodata=nodata,
        )

    def _render(self, file):
        area = self._get_preprocessed_array("area", self.dataset["active"])
        svat = np.arange(1, area.size + 1)

        # Produce values necessary for members without subunit coordinate
        extend_subunits = self.dataset["area"]["subunit"]
        mask = self.dataset["area"].where(self.dataset["active"]).notnull()

        # Generate columns for members without subunit coordinate
        columns = self._get_preprocessed_array(
            "columns", mask, extend_subunits=extend_subunits
        )
        # Generate rows for members without subunit coordinate
        rows = self._get_preprocessed_array(
            "rows", mask, extend_subunits=extend_subunits
        )

        x_coords = self._get_preprocessed_array(
            "x_grid", mask, extend_subunits=extend_subunits
        )

        y_coords = self._get_preprocessed_array(
            "y_grid", mask, extend_subunits=extend_subunits
        )

        # Create DataFrame
        dataframe = pd.DataFrame(
            {
                "svat": svat,
                "rows": rows,
                "columns": columns,
                "y_coords": y_coords,
                "x_coords": x_coords,
            }
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)
