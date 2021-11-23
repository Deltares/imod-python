import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod.msw.pkgbase import MetaData, Package


class GridData(Package):
    """
    This contains the grid data.

    This class is responsible for the file `area_svat.inp`
    """

    _file_name = "area_svat.inp"

    def __init__(
        self,
        area: xr.DataArray,
        landuse: xr.DataArray,
        rootzone_depth: xr.DataArray,
        surface_elevation: xr.DataArray,
        soil_physical_unit: xr.DataArray,
        active: xr.DataArray = None,
    ):
        super().__init__()
        self.dataset["area"] = area
        self.dataset["landuse"] = landuse
        self.dataset["rootzone_depth"] = rootzone_depth
        self.dataset["surface_elevation"] = surface_elevation
        self.dataset["soil_physical_unit"] = soil_physical_unit
        self.dataset["active"] = active

    def _render(self, file):
        # Generate columns for members with subunit coordinate
        area = self._get_preprocessed_array("area", self.dataset["active"])
        landuse = self._get_preprocessed_array(
            "landuse", self.dataset["active"], dtype=int
        )
        rootzone_depth = self._get_preprocessed_array(
            "rootzone_depth", self.dataset["active"]
        )

        # Produce values necessary for members without subunit coordinate
        extend_subunits = self.dataset["area"]["subunit"]
        mask = self._apply_mask(self.dataset["area"], self.dataset["active"]).notnull()

        # Generate columns for members without subunit coordinate
        surface_elevation = self._get_preprocessed_array(
            "surface_elevation", mask, extend_subunits=extend_subunits
        )
        soil_physical_unit = self._get_preprocessed_array(
            "soil_physical_unit",
            mask,
            dtype=int,
            extend_subunits=extend_subunits,
        )

        # Generate remaining columns
        svat = np.arange(1, area.size + 1)
        temp = pd.Series(["" for _ in range(area.size)], dtype="string")
        soil_physical_unit_string = pd.Series(
            ["" for _ in range(area.size)], dtype="string"
        )

        # Create DataFrame
        dataframe = pd.DataFrame(
            {
                "svat": svat,
                "area": area,
                "surface_elevation": surface_elevation,
                "temp": temp,
                "soil_physical_unit": soil_physical_unit,
                "soil_physical_unit_string": soil_physical_unit_string,
                "landuse": landuse,
                "rootzone_depth": rootzone_depth,
            }
        )

        metadata_dict = {
            "svat": MetaData(10, 1, 99999999),
            "area": MetaData(10, 0.0, 999999.0),
            "surface_elevation": MetaData(8, -9999.0, 9999.0),
            "temp": MetaData(8, None, None),
            "soil_physical_unit": MetaData(6, 1, 999999),
            "soil_physical_unit_string": MetaData(16, None, None),
            "landuse": MetaData(6, 1, 999999),
            "rootzone_depth": MetaData(8, 0.0, 10.0),
        }

        self._check_range(dataframe, metadata_dict)

        return self.write_dataframe_fixed_width(file, dataframe, metadata_dict)

    @staticmethod
    def _apply_mask(array, mask):
        if mask is not None:
            return array.where(mask)
        else:
            return array

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)

    def _get_preprocessed_array(
        self, varname: str, mask: xr.DataArray, dtype: type = None, extend_subunits=None
    ):
        array = self.dataset[varname]
        if extend_subunits is not None:
            array = array.expand_dims({"subunit": extend_subunits})

        # Apply mask
        array = self._apply_mask(array, mask)

        # Convert to numpy array and flatten it
        array = array.to_numpy().ravel()

        # Remove NaN values
        array = array[~np.isnan(array)]

        # If dtype isn't None, convert to wanted type
        if dtype:
            array = array.astype(dtype)
        else:
            array = array

        return array
