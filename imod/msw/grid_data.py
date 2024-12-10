import numpy as np
import xarray as xr

from imod.mf6 import StructuredDiscretization
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.regrid.regrid_schemes import GridDataRegridMethod
from imod.msw.utilities.imod5_converter import (
    get_cell_area_from_imod5_data,
    get_landuse_from_imod5_data,
    get_rootzone_depth_from_imod5_data,
    is_msw_active_cell,
)
from imod.msw.utilities.mask import MetaSwapActive, mask_and_broadcast_pkg_data
from imod.typing import Imod5DataDict
from imod.util.spatial import get_cell_area, spatial_reference


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
        imod5_data: Imod5DataDict,
        target_dis: StructuredDiscretization,
    ) -> tuple["GridData", MetaSwapActive]:
        """
        Construct a MetaSWAP GridData package from iMOD5 data in the CAP
        package, loaded with the :func:`imod.formats.prj.open_projectfile_data`
        function.

        Method does the following things:
        - Computes rural area from the wetted area and urban area grids.
        - Sets a second subunit for urban landuse (== 18).
        - Rootzone depth is converted from cm's to m's.
        - Determines where MetaSWAP should be active based on area grids,
          boundary grid, and MODFLOW6 idomain.

        Parameters
        ----------
        imod5_data: Imod5DataDict
            iMOD5 data as returned by
            :func:`imod.formats.prj.open_projectfile_data`
        target_dis: imod.mf6.StructuredDiscretization
            MODFLOW6 discretization to couple the MetaSWAP model to.

        Returns
        -------
        imod.msw.GridData
            GridData package
        MetaSwapActive
            Dataclass containing where MetaSwAP is active, per subunit, as well
            as aggregated over subunits.
        """
        imod5_cap = imod5_data["cap"]

        data = {}
        data["area"] = get_cell_area_from_imod5_data(imod5_cap)
        data["landuse"] = get_landuse_from_imod5_data(imod5_cap)
        data["rootzone_depth"] = get_rootzone_depth_from_imod5_data(imod5_cap)
        data["surface_elevation"] = imod5_cap["surface_elevation"]
        data["soil_physical_unit"] = imod5_cap["soil_physical_unit"].astype(int)

        msw_active = is_msw_active_cell(target_dis, imod5_cap, data["area"])
        data_active = mask_and_broadcast_pkg_data(cls, data, msw_active)
        data_active["active"] = msw_active.all
        return cls(**data_active), msw_active
