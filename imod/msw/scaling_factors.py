import pathlib

import numpy as np
import pandas as pd

from imod.fixed_format import VariableMetaData
from imod.msw.pkgbase import Package


class ScalingFactors(Package):
    """
    This package allows you to do three things:
        1. Set scaling factors for some inputs in the soil physical database,
           namely the soil moisture content and the saturated hydraulic
           conductivity.
        2. Set a scaling factor for pressure head related parameters in the
           landuse class lookup table (LUSE_SVAT.INP).
        3. Set the depth of the perched watertable base.

    This class is useful for sensitivity and uncertainty analyses, as well as
    model calibration. Scaling factors are multiplied with their corresponding
    parameters in the soil physical database.

    Parameters
    ----------
    scale_soil_moisture: array of floats (xr.DataArray)
        Scaling factor which adjusts the saturated soil moisture content, the
        residual soil moisture content, and the soil moisture content of
        macropores. This array must have a subunit coordinate to describe
        different landuses.
    scale_hydraulic_conductivity: array of floats (xr.DataArray)
        Scaling factor which adjusts the (vertical) saturated hydraulic
        conductivity of the soil. This array must have a subunit coordinate to describe
        different landuses.
    scale_pressure_head: array of floats (xr.DataArray)
        Scaling factor which adjusts the pressure head applied to the pressure
        parameters defined in LUSE_SVAT.INP. This array must have a subunit coordinate to describe
        different landuses.
    depth_perched_water_table: array of floats (xr.DataArray)
        Sets the depth of the perched watertable base. If the groundwater depth
        exeeds this depth, the capillary rise is set to zero. This option has
        been included in the model on the request of a specific project (MIPWA),
        and is only sound for depths exceeding 2 meters. For more shallow
        presences of loam causing a perched watertable, it is advised to
        generate a new soil physical unit. This array must not have a subunit
        coordinate.
    active: array of booleans (xr.DataArray)
        Describes whether SVAT units are active or not. This array must not have
        a subunit coordinate.
    """

    _file_name = "uscl_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "scale_soil_moisture": VariableMetaData(8, 0.1, 10.0, float),
        "scale_hydraulic_conductivity": VariableMetaData(8, 0.1, 10.0, float),
        "scale_pressure_head": VariableMetaData(8, 0.1, 10.0, float),
        "depth_perched_water_table": VariableMetaData(8, 0.1, 10.0, float),
    }

    def __init__(
        self,
        scale_soil_moisture,
        scale_hydraulic_conductivity,
        scale_pressure_head,
        depth_perched_water_table,
        active,
    ):
        super().__init__()
        self.dataset["scale_soil_moisture"] = scale_soil_moisture
        self.dataset["scale_hydraulic_conductivity"] = scale_hydraulic_conductivity
        self.dataset["scale_pressure_head"] = scale_pressure_head
        self.dataset["depth_perched_water_table"] = depth_perched_water_table
        self.dataset["active"] = active

    def _render(self, file):
        # Generate columns for members with subunit coordinate
        scale_soil_moisture = self._get_preprocessed_array(
            "scale_soil_moisture", self.dataset["active"]
        )

        scale_hydraulic_conductivity = self._get_preprocessed_array(
            "scale_hydraulic_conductivity", self.dataset["active"]
        )

        scale_pressure_head = self._get_preprocessed_array(
            "scale_pressure_head", self.dataset["active"]
        )

        # Produce values necessary for members without subunit coordinate
        extend_subunits = self.dataset["scale_soil_moisture"]["subunit"]
        mask = (
            self.dataset["scale_soil_moisture"].where(self.dataset["active"]).notnull()
        )

        # Generate columns for members without subunit coordinate
        depth_perched_water_table = self._get_preprocessed_array(
            "depth_perched_water_table", mask, extend_subunits=extend_subunits
        )

        # Generate remaining columns
        svat = np.arange(1, scale_soil_moisture.size + 1)

        # Create DataFrame
        dataframe = pd.DataFrame(
            {
                "svat": svat,
                "scale_soil_moisture": scale_soil_moisture,
                "scale_hydraulic_conductivity": scale_hydraulic_conductivity,
                "scale_pressure_head": scale_pressure_head,
                "depth_perched_water_table ": depth_perched_water_table,
            }
        )

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)
