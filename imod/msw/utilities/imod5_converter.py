from xarray.core.utils import is_scalar

from imod.common.constants import MaskValues
from imod.common.utilities.dataclass_type import DataclassType
from imod.common.utilities.regrid import _regrid_package_data
from imod.logging import LogLevel, logger
from imod.mf6 import StructuredDiscretization
from imod.msw.utilities.common import concat_imod5
from imod.msw.utilities.mask import MetaSwapActive
from imod.typing import GridDataArray, GridDataDict, Imod5DataDict
from imod.typing.grid import ones_like
from imod.util.dims import drop_layer_dim_cap_data
from imod.util.regrid import RegridderWeightsCache
from imod.util.spatial import get_cell_area


def get_cell_area_from_imod5_data(
    imod5_cap: GridDataDict,
) -> GridDataArray:
    # area's per type of svats
    mf6_area = get_cell_area(imod5_cap["boundary"])
    wetted_area = imod5_cap["wetted_area"]
    urban_area = imod5_cap["urban_area"]
    rural_area = mf6_area - (wetted_area + urban_area)
    if (wetted_area > mf6_area).any():
        logger.log(
            loglevel=LogLevel.WARNING,
            message=f"wetted area was set to the max cell area of {mf6_area}",
            additional_depth=0,
        )
        wetted_area = wetted_area.where(wetted_area <= mf6_area, other=mf6_area)
    if (rural_area < 0.0).any():
        logger.log(
            loglevel=LogLevel.WARNING,
            message="found urban area > than (cel-area - wetted area). Urban area was set to 0",
            additional_depth=0,
        )
        urban_area = urban_area.where(rural_area > 0.0, other=0.0)
    rural_area = mf6_area - (wetted_area + urban_area)
    return concat_imod5(rural_area, urban_area)


def get_landuse_from_imod5_data(
    imod5_cap: GridDataDict,
) -> GridDataArray:
    """
    Get landuse from imod5 capillary zone data. This adds two subunits, one
    based on the landuse grid, which specifies rural landuse. The other
    specifies urban landuse, which is coded to value 18.
    """
    rural_landuse = imod5_cap["landuse"]
    # Urban landuse = 18
    urban_landuse = ones_like(rural_landuse) * 18
    return concat_imod5(rural_landuse, urban_landuse).astype(int)


def get_rootzone_depth_from_imod5_data(
    imod5_cap: GridDataDict,
) -> GridDataArray:
    """
    Get rootzone depth from imod5 capillary zone data. Also does a unit
    conversion: iMOD5 specifies rootzone thickness in centimeters, whereas
    MetaSWAP requires rootzone depth in meters.
    """
    rootzone_thickness = imod5_cap["rootzone_thickness"] * 0.01
    # rootzone depth is equal for both svats.
    return concat_imod5(rootzone_thickness, rootzone_thickness)


def is_msw_active_cell(
    target_dis: StructuredDiscretization,
    imod5_cap: GridDataDict,
    msw_area: GridDataArray,
) -> MetaSwapActive:
    """
    Return grid of cells that are active in the coupled computation, based on
    following criteria:

    - Active in top layer MODFLOW6
    - Active in boundary array in CAP package
    - MetaSWAP area > 0

    Returns
    -------
    active: xr.DataArray
        Active cells in any of the subunits
    subunit_active: xr.DataArray
        Cells active per subunit
    """
    mf6_top_active = target_dis["idomain"].isel(layer=0, drop=True)
    subunit_active = (imod5_cap["boundary"] > 0) & (msw_area > 0) & (mf6_top_active > 0)
    active = subunit_active.any(dim="subunit")
    return MetaSwapActive(active, subunit_active)


def _is_equal_scalar_value(da, value):
    """
    Helper function to guarantee that the check in ``has_active_scaling_factor``
    can shortcut after is_scalar returns False.
    """
    return da.to_numpy()[()] == value


def has_active_scaling_factor(imod5_cap: GridDataDict):
    """
    Check if scaling factor grids are active. Carefully checks if data is
    provided as constant (scalar) and if it matches an inactivity value. The
    function shortcuts if data is provided as constant.
    """
    variable_inactive_mapping = {
        "perched_water_table_level": MaskValues.msw_default,
        "soil_moisture_fraction": 1.0,
        "conductivity_factor": 1.0,
    }
    scaling_factor_inactive = True
    for var, inactive_value in variable_inactive_mapping.items():
        da = imod5_cap[var]
        scaling_factor_inactive &= is_scalar(da) and _is_equal_scalar_value(
            da, inactive_value
        )

    return not scaling_factor_inactive


def regrid_imod5_data(
    imod5_data: Imod5DataDict,
    target_dis: StructuredDiscretization,
    regridder_types: DataclassType,
    regrid_cache: RegridderWeightsCache,
) -> Imod5DataDict:
    """
    Regrid iMOD5 CAP data to consistent grid. This is necessary to be able to
    use iMOD5 data in MetaSWAP, as the grid of the iMOD5 CAP data is not
    necessarily the same as the grid of the target MODFLOW6 discretization. The
    regridding process ensures consistency between the iMOD5 CAP data and the
    target MODFLOW6 grid.
    """
    # Drop layer coords
    imod5_cap_no_layer = drop_layer_dim_cap_data(imod5_data)
    target_grid = target_dis.dataset["idomain"].isel(layer=0, drop=True)
    # Regrid the input data
    cap_data_regridded = _regrid_package_data(
        imod5_cap_no_layer["cap"], target_grid, regridder_types, regrid_cache
    )
    extra_paths = imod5_data["extra"]["paths"]
    imod5_regridded: Imod5DataDict = {
        "cap": cap_data_regridded,
        "extra": {"paths": extra_paths},
    }
    return imod5_regridded
