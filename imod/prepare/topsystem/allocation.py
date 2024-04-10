"""
This module contains all kinds of utilities to prepare rivers
"""

from enum import Enum
from typing import Optional

from imod.prepare.layer import (
    create_layered_top,
    get_upper_active_grid_cells,
    get_upper_active_layer_number,
)
from imod.schemata import DimsSchema
from imod.typing import GridDataArray
from imod.util.dims import enforced_dim_order


class ALLOCATION_OPTION(Enum):
    """
    Enumerator for allocation settings. Numbers match the IDEFLAYER options in
    iMOD 5.6.
    """

    stage_to_riv_bot = 0
    first_active_to_riv_bot = -1
    first_active_to_riv_bot__drn = 1
    at_elevation = 2
    at_first_active = 9  # Not an iMOD 5.6 option


PLANAR_GRID = (
    DimsSchema("time", "y", "x")
    | DimsSchema("y", "x")
    | DimsSchema("time", "{face_dim}")
    | DimsSchema("{face_dim}")
)


def allocate_riv_cells(
    allocation_option: ALLOCATION_OPTION,
    active: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
) -> tuple[GridDataArray, Optional[GridDataArray]]:
    match allocation_option:
        case ALLOCATION_OPTION.stage_to_riv_bot:
            return _allocate_cells__stage_to_riv_bot(
                top, bottom, stage, bottom_elevation
            )
        case ALLOCATION_OPTION.first_active_to_riv_bot:
            return _allocate_cells__first_active_to_riv_bot(
                active, top, bottom, bottom_elevation
            )
        case ALLOCATION_OPTION.first_active_to_riv_bot__drn:
            return _allocate_cells__first_active_to_riv_bot__drn(
                active, top, bottom, stage, bottom_elevation
            )
        case ALLOCATION_OPTION.at_elevation:
            return _allocate_cells__at_elevation(top, bottom, bottom_elevation)
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active)
        case _:
            raise ValueError(
                "Received incompatible setting for rivers, only"
                f"'{ALLOCATION_OPTION.stage_to_riv_bot.name}' and"
                f"'{ALLOCATION_OPTION.first_active_to_riv_bot.name}' and"
                f"'{ALLOCATION_OPTION.first_active_to_riv_bot__drn.name}' and"
                f"'{ALLOCATION_OPTION.at_elevation.name}' and"
                f"'{ALLOCATION_OPTION.at_first_active.name}' supported."
                f"got: '{allocation_option.name}'"
            )


def allocate_drn_cells(
    allocation_option: ALLOCATION_OPTION,
    active: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    elevation: GridDataArray,
) -> GridDataArray:
    match allocation_option:
        case ALLOCATION_OPTION.at_elevation:
            return _allocate_cells__at_elevation(top, bottom, elevation)[0]
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active)[0]
        case _:
            raise ValueError(
                "Received incompatible setting for drains, only"
                f"'{ALLOCATION_OPTION.at_elevation.name}' and"
                f"'{ALLOCATION_OPTION.at_first_active.name}' supported."
                f"got: '{allocation_option.name}'"
            )


def allocate_ghb_cells(
    allocation_option: ALLOCATION_OPTION,
    active: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    head: GridDataArray,
) -> GridDataArray:
    match allocation_option:
        case ALLOCATION_OPTION.at_elevation:
            return _allocate_cells__at_elevation(top, bottom, head)[0]
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active)[0]
        case _:
            raise ValueError(
                "Received incompatible setting for general head boundary, only"
                f"'{ALLOCATION_OPTION.at_elevation.name}' and"
                f"'{ALLOCATION_OPTION.at_first_active.name}' supported."
                f"got: '{allocation_option.name}'"
            )


def allocate_rch_cells(
    allocation_option: ALLOCATION_OPTION,
    active: GridDataArray,
) -> GridDataArray:
    match allocation_option:
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active)[0]
        case _:
            raise ValueError(
                "Received incompatible setting for recharge, only"
                f"'{ALLOCATION_OPTION.at_first_active.name}' supported."
                f"got: '{allocation_option.name}'"
            )


def _is_layered(grid: GridDataArray):
    return "layer" in grid.sizes and grid.sizes["layer"] > 1


@enforced_dim_order
def _allocate_cells__stage_to_riv_bot(
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
) -> tuple[GridDataArray, None]:
    """
    Allocate cells inbetween river stage and river bottom_elevation. Compared to
    iMOD5.6, this is similar to setting IDEFFLAYER=0 in the RUNFILE function.

    Note that ``stage`` and ``bottom_elevation`` are not allowed to have a layer
    dimension.

    Parameters
    ----------
    top: GridDataArray
        top of model layers
    bottom: GridDataArray
        bottom of model layers
    stage: GridDataArray
        river stage
    bottom_elevation: GridDataArray
        river bottom elevation

    Returns
    -------
    GridDataArray
        River cells
    """
    PLANAR_GRID.validate(stage)
    PLANAR_GRID.validate(bottom_elevation)

    if _is_layered(top):
        top_layered = top
    else:
        top_layered = create_layered_top(bottom, top)

    riv_cells = (stage > bottom) & (bottom_elevation < top_layered)

    return riv_cells, None


@enforced_dim_order
def _allocate_cells__first_active_to_riv_bot(
    active: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    bottom_elevation: GridDataArray,
) -> tuple[GridDataArray, None]:
    """
    Allocate cells inbetween first active layer and river bottom elevation.
    Compared to iMOD5.6, this is similar to setting IDEFFLAYER=-1 in the RUNFILE
    function.

    Note that ``bottom_elevation`` is not allowed to have a layer dimension.

    Parameters
    ----------
    active: GridDataArray
        active model cells
    top: GridDataArray
        top of model layers
    bottom: GridDataArray
        bottom of model layers
    bottom_elevation: GridDataArray
        river bottom elevation

    Returns
    -------
    GridDataArray
        River cells
    """
    PLANAR_GRID.validate(bottom_elevation)

    upper_active_layer = get_upper_active_layer_number(active)
    layer = active.coords["layer"]

    if _is_layered(top):
        top_layered = top
    else:
        top_layered = create_layered_top(bottom, top)

    riv_cells = (upper_active_layer <= layer) & (bottom_elevation < top_layered)

    return riv_cells, None


@enforced_dim_order
def _allocate_cells__first_active_to_riv_bot__drn(
    active: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
) -> tuple[GridDataArray, GridDataArray]:
    """
    Allocate cells inbetween first active layer and river bottom elevation.
    Cells above river stage are deactivated and returned as drn cells. Compared
    to iMOD5.6, this is similar to setting IDEFFLAYER=1 in the RUNFILE function.

    Note that ``stage`` and ``bottom_elevation`` are not allowed to have a layer
    dimension.

    Parameters
    ----------
    active: GridDataArray
        active grid cells
    top: GridDataArray
        top of model layers
    bottom: GridDataArray
        bottom of model layers
    stage: GridDataArray
        river stage
    bottom_elevation: GridDataArray
        river bottom elevation

    Returns
    -------
    riv_cells: GridDataArray
        River cells (below stage)
    drn_cells: GridDataArray
        Drainage cells (above stage)
    """

    PLANAR_GRID.validate(stage)
    PLANAR_GRID.validate(bottom_elevation)

    if _is_layered(top):
        top_layered = top
    else:
        top_layered = create_layered_top(bottom, top)

    upper_active_layer = get_upper_active_layer_number(active)
    layer = active.coords["layer"]
    drn_cells = (upper_active_layer <= layer) & (bottom >= stage)
    riv_cells = (
        (upper_active_layer <= layer) & (bottom_elevation < top_layered)
    ) != drn_cells

    return riv_cells, drn_cells


@enforced_dim_order
def _allocate_cells__at_elevation(
    top: GridDataArray, bottom: GridDataArray, elevation: GridDataArray
) -> tuple[GridDataArray, None]:
    """
    Allocate cells in river bottom elevation layer. Compared to iMOD5.6, this is
    similar to setting IDEFFLAYER=2 in the RUNFILE function.

    Note that ``bottom_elevation`` is not allowed to have a layer dimension.

    Parameters
    ----------
    top: GridDataArray
        top of model layers
    bottom: GridDataArray
        bottom of model layers
    elevation: GridDataArray
        elevation. Can be river bottom, drain elevation or head of GHB.

    Returns
    -------
    GridDataArray
        River cells
    """

    PLANAR_GRID.validate(elevation)

    if _is_layered(top):
        top_layered = top
    else:
        top_layered = create_layered_top(bottom, top)

    riv_cells = (elevation < top_layered) & (elevation >= bottom)

    return riv_cells, None


@enforced_dim_order
def _allocate_cells__at_first_active(
    active: GridDataArray,
) -> tuple[GridDataArray, None]:
    """
    Allocate cells inbetween first active layer and river bottom elevation.
    Compared to iMOD5.6, this is similar to setting IDEFFLAYER=-1 in the RUNFILE
    function.

    Note that ``bottom_elevation`` is not allowed to have a layer dimension.

    Parameters
    ----------
    active: GridDataArray
        active model cells

    Returns
    -------
    GridDataArray
        River cells
    """

    return get_upper_active_grid_cells(active), None
