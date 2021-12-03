import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod.msw.pkgbase import Package, VariableMetaData


class GridData(Package):
    """
    This contains the grid data.

    This class is responsible for the file `area_svat.inp`
    """

    _file_name = "area_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "area": VariableMetaData(10, 0.0, 999999.0, float),
        "surface_elevation": VariableMetaData(8, -9999.0, 9999.0, float),
        "temp": VariableMetaData(8, None, None, str),
        "soil_physical_unit": VariableMetaData(6, 1, 999999, int),
        "soil_physical_unit_string": VariableMetaData(16, None, None, str),
        "landuse": VariableMetaData(6, 1, 999999, int),
        "rootzone_depth": VariableMetaData(8, 0.0, 10.0, float),
    }

    def __init__(
        self,
        area: xr.DataArray,
        landuse: xr.DataArray,
        rootzone_depth: xr.DataArray,
        surface_elevation: xr.DataArray,
        soil_physical_unit: xr.DataArray,
        active: xr.DataArray,
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
        landuse = self._get_preprocessed_array("landuse", self.dataset["active"])
        rootzone_depth = self._get_preprocessed_array(
            "rootzone_depth", self.dataset["active"]
        )

        # Produce values necessary for members without subunit coordinate
        extend_subunits = self.dataset["area"]["subunit"]
        mask = self.dataset["area"].where(self.dataset["active"]).notnull()

        # Generate columns for members without subunit coordinate
        surface_elevation = self._get_preprocessed_array(
            "surface_elevation", mask, extend_subunits=extend_subunits
        )
        soil_physical_unit = self._get_preprocessed_array(
            "soil_physical_unit",
            mask,
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

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)
