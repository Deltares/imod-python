from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional, TextIO

import cftime
import numpy as np
import pandas as pd
import xarray as xr

import imod
from imod.common.utilities.clip import clip_spatial_box, clip_time_slice
from imod.common.utilities.dataclass_type import RegridMethodType
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.utilities.common import find_in_file_list
from imod.prepare import common
from imod.typing import GridDataArray, Imod5DataDict, IntArray
from imod.util.regrid import RegridderWeightsCache


def _is_parsable_and_existing_path(potential_path: str, mete_grid_path: Path) -> bool:
    """
    mete_grid.inp can contain values like "0.", which are converted to float by
    MetaSWAP. String is converted to path and checked if existing path.
    """
    try:
        float(potential_path)
        return False
    except ValueError:
        # Resolve paths relative to mete_grid.inp path.
        path = mete_grid_path / ".." / Path(potential_path)
        return path.is_file()


def open_first_meteo_grid(mete_grid_path: str | Path, column_nr: int) -> xr.DataArray:
    """
    Find and open first meteo grid path in mete_grid.inp. This grid is enough to
    generate meteomappings. There can be floats before in the column which
    should be skipped.
    """
    if column_nr not in [2, 3]:
        raise ValueError("Column nr should be 2 or 3")

    mete_grid_path = Path(mete_grid_path)
    with open(mete_grid_path, "r") as f:
        lines = f.readlines()

    potential_paths = [line.split(",")[column_nr].replace('"', "") for line in lines]
    for potential_path in potential_paths:
        if _is_parsable_and_existing_path(potential_path, mete_grid_path):
            resolved_path = mete_grid_path / ".." / Path(potential_path)
            return imod.rasterio.open(resolved_path)

    error_message = dedent(f"""    
    Did not find parsable path to existing .ASC file in column {column_nr}. Got
    values (printing first 10): {potential_paths[:10]}.""")

    raise ValueError(error_message)


def open_first_meteo_grid_from_imod5_data(imod5_data: Imod5DataDict, column_nr: int):
    paths = imod5_data["extra"]["paths"]
    metegrid_path = find_in_file_list("mete_grid.inp", paths)
    return open_first_meteo_grid(metegrid_path, column_nr=column_nr)


class MeteoMapping(MetaSwapPackage):
    """
    This class provides common methods for creating mappings between
    meteorological data and MetaSWAP grids. It should not be instantiated
    by the user but rather be inherited from within imod-python to create
    new packages.
    """

    def __init__(self, meteo_grid: GridDataArray):
        super().__init__()
        self.meteo = meteo_grid

    def _render(
        self,
        file: TextIO,
        index: IntArray,
        svat: xr.DataArray,
        *args: Any,
    ):
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
        flip_meteo_x = meteo_grid.indexes["x"].is_monotonic_decreasing
        flip_meteo_y = meteo_grid.indexes["y"].is_monotonic_decreasing
        nrow = meteo_grid["y"].size
        ncol = meteo_grid["x"].size

        # Convert to cell boundaries for the meteo grid
        # Method always returns monotonic increasing edges
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

        # Flip axis when meteofile bound are flipped, relative to the coords
        if flip_meteo_y:
            row = (nrow + 1) - row
        if flip_meteo_x:
            column = (ncol + 1) - column

        n_subunit = svat["subunit"].size

        return np.tile(row, n_subunit), np.tile(column, n_subunit)

    def regrid_like(
        self,
        target_grid: GridDataArray,
        regrid_context: RegridderWeightsCache,
        regridder_types: Optional[RegridMethodType] = None,
    ):
        return deepcopy(self)

    def clip_box(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ):
        """Clip meteo grid to a box defined by time and space."""
        selection = self.meteo.to_dataset(name="meteo")  # Force to dataset
        selection = clip_time_slice(selection, time_min=time_min, time_max=time_max)
        selection = clip_spatial_box(
            selection,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        cls = type(self)
        return cls(selection["meteo"])


class PrecipitationMapping(MeteoMapping):
    """
    This contains the data to connect precipitation grid cells to MetaSWAP
    svats. The precipitation grid does not have to be equal to the metaswap
    grid: connections between the precipitation cells to svats will be
    established using a nearest neighbour lookup.

    This class is responsible for the file `svat2precgrid.inp`.

    Parameters
    ----------
    precipitation: array of floats (xr.DataArray)
        Describes the precipitation data. The extend of the grid must be larger
        than the MetaSvap grid. The data must also be coarser than the MetaSvap
        grid.
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
        super().__init__(precipitation)

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict) -> "PrecipitationMapping":
        """
        Construct a MetaSWAP PrecipitationMapping package from iMOD5 data in the
        CAP package, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        Opens first ascii grid in mete_grid.inp, which is used to construct
        mappings to svats. The grids should not change in dimension over time.
        No checks are done whether cells switch from inactive to active or vice
        versa.

        Parameters
        ----------
        imod5_data: Imod5DataDict
            iMOD5 data as returned by
            :func:`imod.formats.prj.open_projectfile_data`

        Returns
        -------
        imod.msw.PrecipitationMapping
        """
        column_nr = 2
        meteo_grid = open_first_meteo_grid_from_imod5_data(imod5_data, column_nr)
        return cls(meteo_grid)


class EvapotranspirationMapping(MeteoMapping):
    """
    This contains the data to connect evapotranspiration grid cells to MetaSWAP
    svats. The evapotranspiration grid does not have to be equal to the metaswap
    grid: connections between the evapotranspiration cells to svats will be
    established using a nearest neighbour lookup.

    This class is responsible for the file `svat2etrefgrid.inp`.

    Parameters
    ----------
    evapotransporation: array of floats (xr.DataArray)
        Describes the evapotransporation data. The extend of the grid must be
        larger than the MetaSvap grid. The data must also be coarser than the
        MetaSvap grid.
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
        super().__init__(evapotranspiration)

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict) -> "EvapotranspirationMapping":
        """
        Construct a MetaSWAP EvapotranspirationMapping package from iMOD5 data
        in the CAP package, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        Opens first ascii grid in mete_grid.inp, which is used to construct
        mappings to svats. The grids should not change in dimension over time.
        No checks are done whether cells switch from inactive to active or vice
        versa.

        Parameters
        ----------
        imod5_data: Imod5DataDict
            iMOD5 data as returned by
            :func:`imod.formats.prj.open_projectfile_data`

        Returns
        -------
        imod.msw.EvapotranspirationMapping
        """
        column_nr = 3
        meteo_grid = open_first_meteo_grid_from_imod5_data(imod5_data, column_nr)
        return cls(meteo_grid)
