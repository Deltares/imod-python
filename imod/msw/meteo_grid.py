import csv
from pathlib import Path
from shutil import copyfile
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

import imod
from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.common.utilities.regrid_method_type import EmptyRegridMethod, RegridMethodType
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.regrid.regrid_schemes import MeteoGridRegridMethod
from imod.msw.timeutil import to_metaswap_timeformat
from imod.msw.utilities.common import find_in_file_list
from imod.msw.utilities.mask import MaskValues
from imod.typing import Imod5DataDict


class MeteoGrid(MetaSwapPackage, IRegridPackage):
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

    _regrid_method = MeteoGridRegridMethod()

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
        return str(directory / imod.util.path.compose(d, pattern))

    def _is_grid(self, varname: str):
        coords = self.dataset[varname].coords

        if "y" not in coords and "x" not in coords:
            return False
        else:
            return True

    def _compose_dataframe(self, times: np.ndarray):
        dataframe = pd.DataFrame(index=times)

        year, time_since_start_year = to_metaswap_timeformat(times)

        dataframe["time_since_start_year"] = time_since_start_year
        dataframe["year"] = year

        # Data dir is always relative to model dir, so don't use model directory
        # here
        data_dir = Path(".") / self._meteo_dirname

        for varname in self.dataset.data_vars:
            # If grid, we have to add the filename of the .asc to be written
            if self._is_grid(str(varname)):
                dataframe[varname] = [
                    self._compose_filename(
                        {"time": time, "name": varname, "extension": ".asc"},
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

    def write(self, directory: Union[str, Path], *args):
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
            if self._is_grid(str(varname)):
                path = (directory / self._meteo_dirname / str(varname)).with_suffix(
                    ".asc"
                )
                imod.rasterio.save(
                    path, self.dataset[str(varname)], nodata=MaskValues.default
                )

    def _pkgcheck(self):
        for varname in self.dataset.data_vars:
            coords = self.dataset[varname].coords
            if "time" not in coords:
                raise ValueError(f"No 'time' coordinate included in {varname}")

            allowed_dims = ["time", "y", "x"]

            excess_dims = set(self.dataset[varname].dims) - set(allowed_dims)
            if len(excess_dims) > 0:
                raise ValueError(
                    f"Received excess dims {excess_dims} in {self.__class__} for "
                    f"{varname}, please provide data with {allowed_dims}"
                )


class MeteoGridCopy(MetaSwapPackage, IRegridPackage):
    """
    Class to copy existing ``mete_grid.inp``, which contains the meteorological
    grid data. Next to a MeteoGridCopy instance, instances of
    PrecipitationMapping and EvapotranspirationMapping are required as well to
    specify meteorological information to MetaSWAP.

    Parameters
    ----------
    path: Path to mete_grid.inp file
    """

    _file_name = "mete_grid.inp"
    _meteo_dirname = "meteo_grids"

    _regrid_method: RegridMethodType = EmptyRegridMethod()

    def __init__(self, path: Path | str):
        super().__init__()
        self.dataset["path"] = path

    def write(self, directory: Path | str, *args):
        directory = Path(directory)
        path_metegrid = Path(str(self.dataset["path"].values[()]))
        new_path = directory / self._file_name
        copyfile(path_metegrid, new_path)

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict) -> "MeteoGridCopy":
        """
        Construct a MetaSWAP MeteoGridCopy package from iMOD5 data in the CAP
        package, loaded with the :func:`imod.formats.prj.open_projectfile_data`
        function.

        Parameters
        ----------
        imod5_data: Imod5DataDict
            iMOD5 data as returned by
            :func:`imod.formats.prj.open_projectfile_data`

        Returns
        -------
        imod.msw.MeteoGridCopy
        """

        paths = imod5_data["extra"]["paths"]
        filepath = find_in_file_list(cls._file_name, paths)

        return cls(filepath)
