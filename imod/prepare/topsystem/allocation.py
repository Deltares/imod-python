"""
This module contains all kinds of utilities to prepare rivers
"""

from enum import Enum

from imod.mf6.utilities.grid import create_layered_top
from imod.prepare.layer import (
    get_upper_active_grid_cells,
    get_upper_active_layer_number,
)
from imod.schemata import DimsSchema
from imod.typing import GridDataArray


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


def allocate_river_cells(
    allocation_option: ALLOCATION_OPTION,
    active: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
):
    match allocation_option:
        case ALLOCATION_OPTION.stage_to_riv_bot:
            return _allocate_cells__stage_to_riv_bot(
                top, bottom, stage, bottom_elevation
            )
        case ALLOCATION_OPTION.first_active_to_riv_bot:
            return _allocate_cells__first_active_to_riv_bot(
                active, bottom, bottom_elevation
            )
        case ALLOCATION_OPTION.first_active_to_riv_bot__drn:
            return _allocate_cells__first_active_to_riv_bot__drn(
                active, top, bottom, stage, bottom_elevation
            )
        case ALLOCATION_OPTION.at_elevation:
            return _allocate_cells__at_elevation(top, bottom, bottom_elevation)
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active)


def allocate_drain_cells(
    allocation_option: ALLOCATION_OPTION,
    active: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    elevation: GridDataArray,
):
    match allocation_option:
        case ALLOCATION_OPTION.at_elevation:
            return _allocate_cells__at_elevation(top, bottom, elevation)
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active)
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
):
    match allocation_option:
        case ALLOCATION_OPTION.at_elevation:
            return _allocate_cells__at_elevation(top, bottom, head)
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active)
        case _:
            raise ValueError(
                "Received incompatible setting for drains, only"
                f"'{ALLOCATION_OPTION.at_elevation.name}' and"
                f"'{ALLOCATION_OPTION.at_first_active.name}' supported."
                f"got: '{allocation_option.name}'"
            )


def allocate_rch_cells(
    allocation_option: ALLOCATION_OPTION,
    active: GridDataArray,
):
    match allocation_option:
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active)
        case _:
            raise ValueError(
                "Received incompatible setting for drains, only"
                f"'{ALLOCATION_OPTION.at_first_active.name}' supported."
                f"got: '{allocation_option.name}'"
            )


def _is_layered(grid: GridDataArray):
    return "layer" in grid.sizes and grid.sizes["layer"] > 1


def _allocate_cells__stage_to_riv_bot(
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
):
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

    return (stage <= top_layered) & (bottom_elevation >= bottom)


def _allocate_cells__first_active_to_riv_bot(
    active: GridDataArray, bottom: GridDataArray, bottom_elevation: GridDataArray
):
    """
    Allocate cells inbetween first active layer and river bottom elevation.
    Compared to iMOD5.6, this is similar to setting IDEFFLAYER=-1 in the RUNFILE
    function.

    Note that ``bottom_elevation`` is not allowed to have a layer dimension.

    Parameters
    ----------
    active: GridDataArray
        active model cells
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

    return (layer >= upper_active_layer) & (bottom_elevation >= bottom)


def _allocate_cells__first_active_to_riv_bot__drn(
    active: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
):
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
    drn_cells = (layer >= upper_active_layer) & (stage <= top_layered)
    riv_cells = (layer >= upper_active_layer) & (
        bottom_elevation >= bottom
    ) != drn_cells

    return riv_cells, drn_cells


def _allocate_cells__at_elevation(
    top: GridDataArray, bottom: GridDataArray, elevation: GridDataArray
):
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

    return (elevation < top_layered) & (elevation >= bottom)


def _allocate_cells__at_first_active(active: GridDataArray):
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

    return get_upper_active_grid_cells(active)
