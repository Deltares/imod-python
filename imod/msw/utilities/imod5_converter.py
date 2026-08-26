"""
iMOD5 to MetaSWAP converter utilities.
"""

from typing import TypedDict, cast

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.utils import is_scalar

from imod.common.constants import MaskValues
from imod.logging import LogLevel, logger
from imod.mf6 import StructuredDiscretization
from imod.msw.utilities.common import concat_imod5
from imod.msw.utilities.mask import MetaSwapActive
from imod.typing import GridDataArray, GridDataDict, Imod5DataDict
from imod.typing.grid import ones_like, zeros_like
from imod.util.spatial import get_cell_area


# Some additional type aliases for sprinkling data, which is a bit more complex
# than other packages.
class CapSprinklingDataDict(TypedDict, total=False):
    artificial_recharge: GridDataArray
    artificial_recharge_layer: pd.DataFrame
    artificial_recharge_capacity: GridDataArray


class SprinklingPointsDataDict(TypedDict, total=False):
    x_p: np.ndarray | list[float]
    y_p: np.ndarray | list[float]
    layer_p: np.ndarray | list[int]
    id_sprinkling_p: np.ndarray | list[int]
    capacity_p: np.ndarray | list[float]


class SprinklingPointsGridDataDict(SprinklingPointsDataDict, total=False):
    art_grid: GridDataArray


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


def is_sprinkling_from_points(imod5_data: Imod5DataDict) -> bool:
    """
    Check if sprinkling is specified from points, based on the presence of
    sprinkling layer and sprinkling points data in the iMOD5 CAP dataset.
    """
    cap_data = cast(CapSprinklingDataDict, imod5_data["cap"])
    if isinstance(cap_data.get("artificial_recharge_layer"), pd.DataFrame):
        return True
    return False


def sprinkling_data_from_imod5_ipf(
    cap_data: CapSprinklingDataDict,
) -> SprinklingPointsGridDataDict:
    """
    Extract sprinkling data from the iMOD5 CAP dataset that has artificial
    recharge specified in an IPF file and convert it into a format suitable for
    SprinklingPoints

    Parameters
    ----------
    cap_data: CapSprinklingDataDict
        iMOD5 CAP dataset containing artificial recharge grid mapping and
        sprinkling points data

    Returns
    -------
    SprinklingPointsGridDataDict
        Dictionary containing the artificial recharge grid and sprinkling points
        data for SprinklingPoints.
    """
    art_grid = cap_data["artificial_recharge"]
    # Set urban landuse irrigation to 0, as sprinkling is not allowed for urban landuse.
    subunit_template = xr.DataArray(
        np.array([1, 0], dtype=int), dims="subunit", coords={"subunit": [0, 1]}
    )
    art_grid = subunit_template * art_grid

    df_points = cap_data["artificial_recharge_layer"]
    # Select first 5 columns and enforce column names, iMOD5 expects columns in
    # this order. The additional columns are metadata for the user and can be
    # ignored.
    arl_points = df_points.iloc[:, :5]
    arl_points.columns = ["x_p", "y_p", "layer_p", "id_sprinkling_p", "capacity_p"]
    # Enforce dtypes
    dtype_dict = {
        "x_p": float,
        "y_p": float,
        "layer_p": int,
        "id_sprinkling_p": int,
        "capacity_p": float,
    }

    arl_points = arl_points.astype(dtype_dict)
    arl_point_dict = cast(
        SprinklingPointsDataDict,
        {key: arl_points[key].to_numpy() for key in dtype_dict.keys()},
    )

    return {
        "art_grid": art_grid,
        **arl_point_dict,
    }


def sprinkling_data_from_imod5_grid(cap_data: GridDataDict) -> GridDataDict:
    """
    Extract sprinkling data from the iMOD5 CAP dataset that has artificial
    recharge specified on a grid and convert it into a format suitable for
    SprinklingGrid.

    Parameters
    ----------
    cap_data: GridDataDict
        iMOD5 CAP dataset containing artificial recharge and sprinkling layer
        data on a grid.

    Returns
    -------
    GridDataDict
        Dictionary containing the sprinkling capacities grids for
        SprinklingGrid.
    """
    # Convert units from mm/d to m3/d
    msw_area = get_cell_area_from_imod5_data(cap_data)
    capacity_mmd = cap_data["artificial_recharge_capacity"]
    capacity_m3d = capacity_mmd * 1e-3 * msw_area.sel(subunit=0, drop=True)

    artificial_rch_type = cap_data["artificial_recharge"]
    from_groundwater = artificial_rch_type == 1
    from_surfacewater = artificial_rch_type == 2
    is_active = artificial_rch_type != 0

    zero_where_active = zeros_like(artificial_rch_type).where(is_active)

    # Add zero where active, to have active cells set to 0.0.
    max_abstraction_groundwater_rural = zero_where_active.where(
        ~from_groundwater, capacity_m3d
    )
    max_abstraction_surfacewater_rural = zero_where_active.where(
        ~from_surfacewater, capacity_m3d
    )

    # No sprinkling for urban environments
    max_abstraction_urban = zero_where_active

    data = {}
    data["max_abstraction_groundwater"] = concat_imod5(
        max_abstraction_groundwater_rural, max_abstraction_urban
    )
    data["max_abstraction_surfacewater"] = concat_imod5(
        max_abstraction_surfacewater_rural, max_abstraction_urban
    )
    return data
