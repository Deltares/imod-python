import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod.fixed_format import VariableMetaData
from imod.msw.pkgbase import Package


class Infiltration(Package):
    """
    This contains the infiltration data.

    This class is responsible for the file `infi_svat.inp`

    Parameters
    ----------
    infiltration_capacity: array of floats (xr.DataArray)
        Describes the infiltration capacity of SVAT units.
        This array must have a subunit coordinate to describe different land uses.
    downward_resistance: array of floats (xr.DataArray)
        Describes the downward resisitance of SVAT units.
        This array must not have a subunit coordinate.
    upward_resistance: array of floats (xr.DataArray)
        Describes the upward resistance of SVAT units.
        This array must not have a subunit coordinate.
    bottom_resistance: array of floats (xr.DataArray)
        Describes the infiltration capacity of SVAT units.
        This array must not have a subunit coordinate.
    active: array of bools (xr.DataArray)
        Describes whether SVAT units are active or not.
        This array must not have a subunit coordinate.
    """

    _file_name = "infi_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "infiltration_capacity": VariableMetaData(8, 0.0, 1000.0, float),
        "downward_resistance": VariableMetaData(8, -9999.0, 999999.0, float),
        "upward_resistance": VariableMetaData(8, -9999.0, 999999.0, float),
        "bottom_resistance": VariableMetaData(8, -9999.0, 999999.0, float),
        "extra_storage_coefficient": VariableMetaData(8, 0.01, 1.0, float),
    }

    def __init__(
        self,
        infiltration_capacity: xr.DataArray,
        downward_resistance: xr.DataArray,
        upward_resistance: xr.DataArray,
        bottom_resistance: xr.DataArray,
        extra_storage_coefficient: xr.DataArray,
        active: xr.DataArray,
    ):
        super().__init__()
        self.dataset["infiltration_capacity"] = infiltration_capacity
        self.dataset["downward_resistance"] = downward_resistance
        self.dataset["upward_resistance"] = upward_resistance
        self.dataset["bottom_resistance"] = bottom_resistance
        self.dataset["extra_storage_coefficient"] = extra_storage_coefficient
        self.dataset["active"] = active

    def _render(self, file):
        # Generate columns for members with subunit coordinate
        infiltration_capacity = self._get_preprocessed_array(
            "infiltration_capacity", self.dataset["active"]
        )

        # Produce values necessary for members without subunit coordinate
        extend_subunits = self.dataset["infiltration_capacity"]["subunit"]
        mask = (
            self.dataset["infiltration_capacity"]
            .where(self.dataset["active"])
            .notnull()
        )

        # Generate columns for members without subunit coordinate
        downward_resistance = self._get_preprocessed_array(
            "downward_resistance", mask, extend_subunits=extend_subunits
        )
        upward_resistance = self._get_preprocessed_array(
            "upward_resistance", mask, extend_subunits=extend_subunits
        )
        bottom_resistance = self._get_preprocessed_array(
            "bottom_resistance", mask, extend_subunits=extend_subunits
        )
        extra_storage_coefficient = self._get_preprocessed_array(
            "extra_storage_coefficient", mask, extend_subunits=extend_subunits
        )

        # Generate remaining columns
        svat = np.arange(1, infiltration_capacity.size + 1)

        # Create DataFrame
        dataframe = pd.DataFrame(
            {
                "svat": svat,
                "infiltration_capacity": infiltration_capacity,
                "downward_resistance": downward_resistance,
                "upward_resistance": upward_resistance,
                "bottom_resistance": bottom_resistance,
                "extra_storage_coefficient": extra_storage_coefficient,
            }
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)
