from textwrap import dedent

import xarray as xr

from imod.logging import LogLevel, logger
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.regrid.regrid_schemes import InfiltrationRegridMethod
from imod.msw.utilities.common import concat_imod5
from imod.msw.utilities.mask import MaskValues
from imod.typing import GridDataDict, Imod5DataDict
from imod.typing.grid import ones_like


def deactivate_small_resistances_in_data(data: GridDataDict):
    """
    Deactivate cells where resistance smaller than 5 days are set to
    -9999.0.
    """
    message = dedent("""Detected cells with resistances smaller than 5.0 in {var}, set
    to inactive""")

    for var in ["downward_resistance", "upward_resistance"]:
        to_deactivate = data[var] < 5.0
        if to_deactivate.any():
            logger.log(
                loglevel=LogLevel.WARNING,
                message=message.format(var=var),
                additional_depth=1,
            )
            data[var] = data[var].where(~to_deactivate, MaskValues.default)
    return data


class Infiltration(MetaSwapPackage, IRegridPackage):
    """
    This contains the infiltration data.

    This class is responsible for the file `infi_svat.inp`

    Parameters
    ----------
    infiltration_capacity: array of floats (xr.DataArray)
        Describes the infiltration capacity of SVAT units. This array must have
        a subunit coordinate to describe different land uses.
    downward_resistance: array of floats (xr.DataArray)
        Describes the downward resisitance of SVAT units. Set to -9999.0 to make
        MetaSWAP ignore this resistance. This array must have a subunit
        coordinate.
    upward_resistance: array of floats (xr.DataArray)
        Describes the upward resistance of SVAT units. Set to -9999.0 to make
        MetaSWAP ignore this resistance. This array must have a subunit
        coordinate.
    bottom_resistance: array of floats (xr.DataArray)
        Describes the infiltration capacity of SVAT units. Set to -9999.0 to
        make MetaSWAP ignore this resistance. This array must not have a subunit
        coordinate.
    extra_storage_coefficient: array of floats (xr.DataArray)
        Extra storage coefficient of phreatic layer. This array must not have a
        subunit coordinate.
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

    _with_subunit = (
        "infiltration_capacity",
        "downward_resistance",
        "upward_resistance",
    )
    _without_subunit = (
        "bottom_resistance",
        "extra_storage_coefficient",
    )
    _to_fill = ()

    _regrid_method = InfiltrationRegridMethod()

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

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict) -> "Infiltration":
        """
        Construct a MetaSWAP Infiltration package from iMOD5 data in the CAP
        package, loaded with the :func:`imod.formats.prj.open_projectfile_data`
        function.

        Concatenates infiltration_capacity, runon_resistance, and
        runoff_resistance along the subunit dimension. 0 = rural landuse, 1 =
        urban landuse. Resistances smaller than 5 days are deactivated for
        transparency, as MetaSWAP also does this internally.

        Parameters
        ----------
        imod5_data: Imod5DataDict
            iMOD5 data as returned by
            :func:`imod.formats.prj.open_projectfile_data`
        
        Returns
        -------
        imod.msw.Infiltration
        """

        cap_data = imod5_data["cap"]
        data = {}
        # Use runon resistance as downward resistance, and runoff for downward
        # resistance
        key_mapping = {
            "infiltration_capacity": "infiltration_capacity",
            "downward_resistance": "runon_resistance",
            "upward_resistance": "runoff_resistance",
        }
        for var_rename, var_key in key_mapping.items():
            data_ls = [
                cap_data[f"{landuse}_{var_key}"] for landuse in ["rural", "urban"]
            ]
            data[var_rename] = concat_imod5(*data_ls)

        data = deactivate_small_resistances_in_data(data)

        like = data["downward_resistance"].isel(subunit=0, drop=True)
        data["bottom_resistance"] = ones_like(like) * MaskValues.default
        data["extra_storage_coefficient"] = ones_like(like)

        return cls(**data)
