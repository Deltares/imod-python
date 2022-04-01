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
    extra_storage_coefficient: array of floats (xr.DataArray)
        Extra storage coefficient of phreatic layer.
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

    _with_subunit = ["infiltration_capacity"]
    _without_subunit = [
        "downward_resistance",
        "upward_resistance",
        "bottom_resistance",
        "extra_storage_coefficient",
    ]
    _to_fill = []

    def __init__(
        self,
        infiltration_capacity: xr.DataArray,
        downward_resistance: xr.DataArray,
        upward_resistance: xr.DataArray,
        bottom_resistance: xr.DataArray,
        extra_storage_coefficient: xr.DataArray,
    ):
        super().__init__()
        self.dataset["infiltration_capacity"] = infiltration_capacity
        self.dataset["downward_resistance"] = downward_resistance
        self.dataset["upward_resistance"] = upward_resistance
        self.dataset["bottom_resistance"] = bottom_resistance
        self.dataset["extra_storage_coefficient"] = extra_storage_coefficient

        self._pkgcheck()
