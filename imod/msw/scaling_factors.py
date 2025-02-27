from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.regrid.regrid_schemes import ScalingRegridMethod
from imod.msw.utilities.common import concat_imod5
from imod.typing import Imod5DataDict
from imod.typing.grid import ones_like


class ScalingFactors(MetaSwapPackage, IRegridPackage):
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
    """

    _file_name = "uscl_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "scale_soil_moisture": VariableMetaData(8, 0.1, 10.0, float),
        "scale_hydraulic_conductivity": VariableMetaData(8, 0.1, 10.0, float),
        "scale_pressure_head": VariableMetaData(8, 0.1, 10.0, float),
        "depth_perched_water_table": VariableMetaData(8, 0.1, 10.0, float),
    }

    _with_subunit = (
        "scale_soil_moisture",
        "scale_hydraulic_conductivity",
        "scale_pressure_head",
    )
    _without_subunit = ("depth_perched_water_table",)
    _to_fill = ()

    _regrid_method = ScalingRegridMethod()

    def __init__(
        self,
        scale_soil_moisture,
        scale_hydraulic_conductivity,
        scale_pressure_head,
        depth_perched_water_table,
    ):
        super().__init__()
        self.dataset["scale_soil_moisture"] = scale_soil_moisture
        self.dataset["scale_hydraulic_conductivity"] = scale_hydraulic_conductivity
        self.dataset["scale_pressure_head"] = scale_pressure_head
        self.dataset["depth_perched_water_table"] = depth_perched_water_table

        self._pkgcheck()

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict) -> "ScalingFactors":
        """
        Construct a MetaSWAP Ponding package from iMOD5 data in the CAP
        package, loaded with the :func:`imod.formats.prj.open_projectfile_data`
        function.

        Pressure head factor is set to one for all subunits. All urban areas
        (subunit=1) are also set to one for all variables, except the perced
        water table level.

        Parameters
        ----------
        imod5_data: Imod5DataDict
            iMOD5 data as returned by
            :func:`imod.formats.prj.open_projectfile_data`

        Returns
        -------
        imod.msw.ScalingFactors
        """
        cap_data = imod5_data["cap"]
        grid_ones = ones_like(cap_data["boundary"])

        data = {}
        data["scale_soil_moisture"] = concat_imod5(
            cap_data["soil_moisture_fraction"], grid_ones
        )
        data["scale_hydraulic_conductivity"] = concat_imod5(
            cap_data["conductivitiy_factor"], grid_ones
        )
        data["scale_pressure_head"] = concat_imod5(grid_ones, grid_ones)
        data["depth_perched_water_table"] = cap_data["perched_water_table_level"]

        return cls(**data)
