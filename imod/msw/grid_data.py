from typing import Optional

import numpy as np
import xarray as xr

from imod.mf6 import StructuredDiscretization
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.regrid.regrid_schemes import (
    RegridMethodType,
)
from imod.mf6.utilities.regrid import RegridderWeightsCache
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.regrid.regrid_schemes import GridDataRegridMethod
from imod.typing import GridDataArray
from imod.typing.grid import concat, ones_like
from imod.util.spatial import get_cell_area, spatial_reference


def _concat_subunits(arg1: GridDataArray, arg2: GridDataArray):
    return concat(
        [arg1, arg2], dim="subunit"
    ).assign_coords(subunit = [0, 1])

def get_cell_area_from_imod5_data(
    imod5_data: dict[str, dict[str, GridDataArray]],
) -> GridDataArray:
    # area's per type of svats
    mf6_area = get_cell_area(imod5_data["cap"]["wetted_area"])
    wetted_area = imod5_data["cap"]["wetted_area"]
    urban_area = imod5_data["cap"]["urban_area"]
    rural_area = mf6_area - (wetted_area + urban_area)
    if (wetted_area > mf6_area).any():
        print(f"wetted area was set to the max cell area of {mf6_area}")
        wetted_area = wetted_area.where(wetted_area <= mf6_area, other=mf6_area)
    if (rural_area < 0.0).any():
        print(
            "found urban area > than (cel-area - wetted area). Urban area was set to 0"
        )
        urban_area = urban_area.where(rural_area > 0.0, other=0.0)
    rural_area = mf6_area - (wetted_area + urban_area)
    return _concat_subunits(rural_area, urban_area)


def get_landuse_from_imod5_data(
    imod5_data: dict[str, dict[str, GridDataArray]],
) -> GridDataArray:
    rural_landuse = imod5_data["cap"]["landuse"]
    # Urban landuse = 18
    urban_landuse = ones_like(rural_landuse) * 18
    return _concat_subunits(rural_landuse, urban_landuse)


def get_rootzone_depth_from_imod5_data(
    imod5_data: dict[str, dict[str, GridDataArray]],
) -> GridDataArray:
    # iMOD5 grid has values in cm,
    # MetaSWAP needs them in m.
    rootzone_thickness = imod5_data["cap"]["rootzone_thickness"] * 0.01
    # rootzone depth is equal for both svats.
    return _concat_subunits(rootzone_thickness, rootzone_thickness)


class GridData(MetaSwapPackage, IRegridPackage):
    """
    This contains the grid data of MetaSWAP.

    This class is responsible for the file `area_svat.inp`

    Parameters
    ----------
    area: array of floats (xr.DataArray)
        Describes the area of SVAT units. This array must have a subunit coordinate
        to describe different landuses.
    landuse: array of integers (xr.DataArray)
        Describes the landuse type of SVAT units.
        This array must have a subunit coordinate.
    rootzone_depth: array of floats (xr.DataArray)
        Describes the rootzone depth of SVAT units.
        This array must have a subunit coordinate to describe different landuses.
    surface_elevation: array of floats (xr.DataArray)
        Describes the surface elevation of SVAT units.
        This array must not have a subunit coordinate.
    soil_physical_unit: array of integers (xr.DataArray)
        Describes the physical parameters of SVAT units.
        These parameters will be looked up in a table according to the given integers.
        This array must not have a subunit coordinate.
    active: array of bools (xr.DataArray)
        Describes whether SVAT units are active or not.
        This array must not have a subunit coordinate.
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
    _with_subunit = ("area", "landuse", "rootzone_depth")
    _without_subunit = ("surface_elevation", "soil_physical_unit")
    _to_fill = ("soil_physical_unit_string", "temp")

    _regrid_method = GridDataRegridMethod()

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

        self._pkgcheck()

    def generate_index_array(self):
        """
        Generate index arrays to be used on other packages
        """
        area = self.dataset["area"]
        active = self.dataset["active"]

        isactive = area.where(active).notnull()

        index = isactive.values.ravel()

        svat = xr.full_like(area, fill_value=0, dtype=np.int64).rename("svat")
        svat.values[isactive.values] = np.arange(1, index.sum() + 1)

        return index, svat

    def _pkgcheck(self):
        super()._pkgcheck()

        dx, _, _, dy, _, _ = spatial_reference(self.dataset)

        if (not np.isscalar(dx)) or (not np.isscalar(dy)):
            raise ValueError("MetaSWAP only supports equidistant grids")

        active = self.dataset["active"]

        cell_area = get_cell_area(active)
        total_area = self.dataset["area"].sum(dim="subunit")

        # Apparently all regional models intentionally provided area grids
        # smaller than cell area, to allow surface waters as workaround.
        unequal_area = (total_area > cell_area).values[active.values]

        if np.any(unequal_area):
            raise ValueError(
                "Provided area grid with total areas larger than cell area"
            )

    @classmethod
    def from_imod5_data(
        cls,
        imod5_data: dict[str, dict[str, GridDataArray]],
        target_dis: StructuredDiscretization,
        regridder_types: Optional[RegridMethodType] = None,
        regrid_cache: RegridderWeightsCache = RegridderWeightsCache(),
    ) -> "GridData":
        data = {}
        data["area"] = get_cell_area_from_imod5_data(imod5_data)
        data["landuse"] = get_landuse_from_imod5_data(imod5_data)
        data["rootzone_depth"] = get_rootzone_depth_from_imod5_data(imod5_data)
        data["surface_elevation"] = imod5_data["cap"]["surface_elevation"]
        data["soil_physical_unit"] = imod5_data["cap"]["soil_physical_unit"]

        mf6_top_active = target_dis["idomain"].isel(layer=0, drop=True)
        subunit_active = (
            (imod5_data["cap"]["boundary"] == 1)
            & (data["area"] > 0)
            & (mf6_top_active >= 1)
        )
        active = subunit_active.all(dim="subunit")
        data_active = {
            key: (
                griddata.where(subunit_active)
                if key in cls._with_subunit
                else griddata.where(active)
            )
            for key, griddata in data.items()
        }
        data_active["active"] = active
        return cls(**data_active)
