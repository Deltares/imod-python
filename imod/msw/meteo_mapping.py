import numpy as np
import pandas as pd
import xarray as xr

from imod.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage
from imod.prepare import common


class MeteoMapping(MetaSwapPackage):
    """
    This class provides common methods for creating mappings between
    meteorological data and MetaSWAP grids. It should not be instantiated
    by the user but rather be inherited from within imod-python to create
    new packages.
    """

    def __init__(self):
        super().__init__()

    def _render(self, file, index, svat):
        data_dict = {"svat": svat.values.ravel()[index]}

        row, column = self.grid_mapping(svat, self.meteo)

        data_dict["row"] = row[index]
        data_dict["column"] = column[index]

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    @staticmethod
    def grid_mapping(svat: xr.DataArray, meteo_grid: xr.DataArray) -> pd.DataFrame:
        flip_svat_x = svat.indexes["x"].is_monotonic_decreasing
        flip_svat_y = svat.indexes["y"].is_monotonic_decreasing
        flip_meteo_x = meteo_grid.indexes["x"].is_monotonic_decreasing
        flip_meteo_y = meteo_grid.indexes["y"].is_monotonic_decreasing
        nrow = meteo_grid["y"].size
        ncol = meteo_grid["x"].size

        # Convert to cell boundaries for the meteo grid
        meteo_x = common._coord(meteo_grid, "x")
        meteo_y = common._coord(meteo_grid, "y")

        # Create the SVAT grid
        svat_grid_y, svat_grid_x = np.meshgrid(svat.y, svat.x, indexing="ij")
        svat_grid_y = svat_grid_y.ravel()
        svat_grid_x = svat_grid_x.ravel()

        # Determine where the svats fit in within the cell boundaries of the meteo grid
        row = np.searchsorted(meteo_y, svat_grid_y)
        column = np.searchsorted(meteo_x, svat_grid_x)

        # Find out of bounds members
        if (column == 0).any() or (column > ncol).any():
            raise ValueError("Some values are out of bounds for column")
        if (row == 0).any() or (row > nrow).any():
            raise ValueError("Some values are out of bounds for row")

        # Flip axis if necessary
        if flip_meteo_y ^ flip_svat_y:
            row = (nrow + 1) - row
        if flip_meteo_x ^ flip_svat_x:
            column = (ncol + 1) - column

        n_subunit, _, _ = svat.shape

        return np.tile(row, n_subunit), np.tile(column, n_subunit)


class PrecipitationMapping(MeteoMapping):
    """
    This contains the data to map precipitation grid cells to MetaSWAP svats.

    This class is responsible for the file `svat2precgrid.inp`.

    Parameters
    ----------
    precipitation: array of floats (xr.DataArray)
        Describes the precipitation data.
        The extend of the grid must be larger than the MetaSvap grid.
        The data must also be coarser than the MetaSvap grid.
    """

    _file_name = "svat2precgrid.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, None, None, int),
        "row": VariableMetaData(10, None, None, int),
        "column": VariableMetaData(10, None, None, int),
    }

    def __init__(
        self,
        precipitation: xr.DataArray,
    ):
        super().__init__()
        self.meteo = precipitation


class EvapotranspirationMapping(MeteoMapping):
    """
    This contains the data to map evapotranspiration grid cells to MetaSWAP svats.

    This class is responsible for the file `svat2etrefgrid.inp`.

    Parameters
    ----------
    evapotransporation: array of floats (xr.DataArray)
        Describes the evapotransporation data.
        The extend of the grid must be larger than the MetaSvap grid.
        The data must also be coarser than the MetaSvap grid.
    """

    _file_name = "svat2etrefgrid.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, None, None, int),
        "row": VariableMetaData(10, None, None, int),
        "column": VariableMetaData(10, None, None, int),
    }

    def __init__(
        self,
        evapotranspiration: xr.DataArray,
    ):
        super().__init__()
        self.meteo = evapotranspiration
