"""
This module contains all kinds of utilities to prepare rivers
"""

from enum import Enum

from imod.mf6.utilities.grid import create_layered_top
from imod.prepare.layer import get_upper_active_layer_number
from imod.schemata import DimsSchema
from imod.typing import GridDataArray


class ALLOCATION_OPTION(Enum):
    """
    Enumerator for allocation settings. Numbers match the IDEFLAYER options in
    iMOD 5.6.
    """
    between_stage_and_riv_bottom = 0
    first_active = -1
    first_active_drn_above_stage = 1
    in_riv_bottom_layer = 2


PLANAR_GRID = (
    DimsSchema("time", "y", "x")
    | DimsSchema("y", "x")
    | DimsSchema("time", "{face_dim}")
    | DimsSchema("{face_dim}")
)


def allocate_cells(
    allocation_option: ALLOCATION_OPTION,
    active: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
):
    match allocation_option:
        case ALLOCATION_OPTION.between_stage_and_riv_bottom:
            return allocate_cells__between_stage_and_bottom_elevation(
                top, bottom, stage, bottom_elevation
            )
        case ALLOCATION_OPTION.first_active:
            return allocate_cells__first_active(active, bottom, bottom_elevation)
        case ALLOCATION_OPTION.first_active_drn_above_stage:
            return allocate_cells__first_active_drn_above_stage(
                active, top, bottom, stage
            )
        case ALLOCATION_OPTION.in_riv_bottom_layer:
            return allocate_cells__in_bottom_elevation_layer(
                top, bottom, bottom_elevation
            )


def _is_layered(grid: GridDataArray):
    return "layer" in grid.sizes and grid.sizes["layer"] > 1


def allocate_cells__between_stage_and_bottom_elevation(
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


def allocate_cells__first_active(
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


def allocate_cells__first_active_drn_above_stage(
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


def allocate_cells__in_bottom_elevation_layer(
    top: GridDataArray, bottom: GridDataArray, bottom_elevation: GridDataArray
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
    bottom_elevation: GridDataArray
        river bottom elevation

    Returns
    -------
    GridDataArray
        River cells
    """

    PLANAR_GRID.validate(bottom_elevation)

    if _is_layered(top):
        top_layered = top
    else:
        top_layered = create_layered_top(bottom, top)

    return (bottom_elevation < top_layered) & (bottom_elevation >= bottom)
