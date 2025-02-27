from xarray.core.utils import is_scalar

from imod.logging import LogLevel, logger
from imod.mf6 import StructuredDiscretization
from imod.msw.utilities.common import concat_imod5
from imod.msw.utilities.mask import MaskValues, MetaSwapActive
from imod.typing import GridDataArray, GridDataDict
from imod.typing.grid import ones_like
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
        "perched_water_table_level": MaskValues.default,
        "soil_moisture_fraction": 1.0,
        "conductivitiy_factor": 1.0,
    }
    scaling_factor_inactive = True
    for var, inactive_value in variable_inactive_mapping.items():
        da = imod5_cap[var]
        scaling_factor_inactive &= is_scalar(da) and _is_equal_scalar_value(
            da, inactive_value
        )

    return not scaling_factor_inactive
