import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from imod.msw.pkgbase import MetaData, Package


class Infiltration(Package):
    """
    This contains the infiltration data.

    This class is responsible for the file `infi_svat.inp`
    """

    _file_name = "infi_svat.inp"

    def __init__(
        self,
        infiltration_capacity: xr.DataArray,
        downward_resistance: xr.DataArray,
        upward_resistance: xr.DataArray,
        bottom_resistance: xr.DataArray,
        extra_storage_coefficient: xr.DataArray,
        active: xr.DataArray = None,
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
        mask = self._apply_mask(
            self.dataset["infiltration_capacity"], self.dataset["active"]
        ).notnull()

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

        metadata_dict = {
            "svat": MetaData(10, 1, 99999999),
            "infiltration_capacity": MetaData(8, 0.0, 1000.0),
            "downward_resistance": MetaData(8, -9999.0, 999999.0),
            "upward_resistance": MetaData(8, -9999.0, 999999.0),
            "bottom_resistance": MetaData(8, -9999.0, 999999.0),
            "extra_storage_coefficient": MetaData(8, 0.01, 1.0),
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
