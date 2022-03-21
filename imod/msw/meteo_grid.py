import csv
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import imod
from imod import util
from imod.msw.pkgbase import Package
from imod.msw.timeutil import to_metaswap_timeformat
from typing import Optional, Union


class MeteoGrid(Package):
    """
    This contains the meteorological grid data. Grids are written to ESRI ASCII
    files. The meteorological data requires a time coordinate. Next to a
    MeteoGrid instance, instances of PrecipitationMapping and
    EvapotranspirationMapping are required as well to specify meteorological
    information to MetaSWAP.

    This class is responsible for `mete_grid.inp`.

    Parameters
    ----------
    precipitation: array of floats (xr.DataArray)
        Contains the precipitation grids in mm/d. A time coordinate is required.
    evapotranspiration: array of floats (xr.DataArray)
        Contains the evapotranspiration grids in mm/d. A time coordinate is
        required.
    """

    _file_name = "mete_grid.inp"
    _meteo_dirname = "meteo_grids"

    def __init__(self, precipitation: xr.DataArray, evapotranspiration: xr.DataArray):
        super().__init__()

        self.dataset["precipitation"] = precipitation
        self.dataset["evapotranspiration"] = evapotranspiration

        self._pkgcheck()

    def write_free_format_file(self, path: Union[str, Path], dataframe: pd.DataFrame):
        """
        Write free format file. The mete_grid.inp file is free format.
        """

        columns = list(self.dataset.data_vars)

        dataframe.loc[:, columns] = '"' + dataframe[columns] + '"'
        # Add required columns, which we will not use.
        # These are only used when WOFOST is used
        # TODO: Add support for temperature to allow WOFOST support
        wofost_columns = [
            "minimum_day_temperature",
            "maximum_day_temperature",
            "mean_temperature",
        ]
        dataframe.loc[:, wofost_columns] = '"NoValue"'

        self.check_string_lengths(dataframe)

        dataframe.to_csv(
            path, header=False, quoting=csv.QUOTE_NONE, float_format="%.4f", index=False
        )

    def _compose_filename(
        self, d: dict, directory: Path, pattern: Optional[str] = None
    ):
        """
        Construct a filename, following the iMOD conventions.


        Parameters
        ----------
        d : dict
            dict of parts (time, layer) for filename.
        pattern : string or re.pattern
            Format to create pattern for.

        Returns
        -------
        str
            Absolute path.

        """
        return str(directory / util.compose(d, pattern))
        # return str(util.compose(d, pattern).resolve())

    def _is_grid(self, varname: str):
        coords = self.dataset[varname].coords

        if "y" not in coords and "x" not in coords:
            return False
        else:
            return True

    def _compose_dataframe(self, times: np.array):
        dataframe = pd.DataFrame(index=times)

        year, time_since_start_year = to_metaswap_timeformat(times)

        dataframe["time_since_start_year"] = time_since_start_year
        dataframe["year"] = year

        # Data dir is always relative to model dir, so don't use model directory
        # here
        data_dir = Path(".") / self._meteo_dirname

        for varname in self.dataset.data_vars:
            # If grid, we have to add the filename of the .asc to be written
            if self._is_grid(varname):
                dataframe[varname] = [
                    self._compose_filename(
                        dict(time=time, name=varname, extension=".asc"),
                        directory=data_dir,
                    )
                    for time in times
                ]
            else:
                dataframe[varname] = self.dataset[varname].values.astype(str)

        return dataframe

    def check_string_lengths(self, dataframe: pd.DataFrame):
        """
        Check if strings lengths do not exceed 256 characters.
        With absolute paths this might be an issue.
        """

        # Because two quote marks are added later.
        character_limit = 254

        columns = list(self.dataset.data_vars)

        str_too_long = [
            np.any(dataframe[varname].str.len() > character_limit)
            for varname in columns
        ]

        if any(str_too_long):
            indexes_true = np.where(str_too_long)[0]
            too_long_columns = list(np.array(columns)[indexes_true])
            raise ValueError(
                f"Encountered strings longer than 256 characters in columns: {too_long_columns}"
            )

    def write(self, directory: Union[str, Path]):
        """
        Write mete_grid.inp and accompanying ASCII grid files.

        Parameters
        ----------
        directory: str or Path
            directory to write file in.
        """

        directory = Path(directory)

        times = self.dataset["time"].values

        dataframe = self._compose_dataframe(times)
        self.write_free_format_file(directory / self._file_name, dataframe)

        # Write grid data to ESRI ASCII files
        for varname in self.dataset.data_vars:
            if self._is_grid(varname):
                path = (directory / self._meteo_dirname / varname).with_suffix(".asc")
                imod.rasterio.save(path, self.dataset[varname], nodata=-9999.0)

    def _pkgcheck(self):
        for varname in self.dataset.data_vars:
            if "time" not in self.dataset[varname].coords:
                raise ValueError(f"No time coordinate included in {varname}")
