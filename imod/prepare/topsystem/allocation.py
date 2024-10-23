"""
This module contains all kinds of utilities to prepare rivers
"""

from enum import Enum
from typing import Optional

import numpy as np

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
    Enumerator for settings to allocate planar grid with RIV, DRN, GHB, or RCH
    cells over the vertical layer dimensions. Numbers match the IDEFLAYER
    options in iMOD 5.6.

    * ``stage_to_riv_bot``: RIV. Allocate cells spanning from the river stage up
      to the river bottom elevation. This matches the iMOD 5.6 IDEFLAYER = 0
      option.
    * ``first_active_to_elevation``: RIV, DRN, GHB. Allocate cells spanning from
      first upper active cell up to the river bottom elevation. This matches the
      iMOD 5.6 IDEFLAYER = -1 option.
    * ``stage_to_riv_bot_drn_above``: RIV. Allocate cells spanning from first
      upper active cell up to the river bottom elevation. Method returns both
      allocated cells for a river package as well as a drain package. Cells
      above river stage are allocated as drain cells, cells below are as river
      cells. This matches the iMOD 5.6 IDEFLAYER = 1 option.
    * ``at_elevation``: RIV, DRN, GHB. Allocate cells containing the river
      bottom elevation, drain elevation, or head respectively for river, drain
      and general head boundary. This matches the iMOD 5.6 IDEFLAYER = 2
      option.
    * ``at_first_active``: RIV, DRN, GHB, RCH. Allocate cells at the upper
      active cells. This has no equivalent option in iMOD 5.6.

    Examples
    --------

    >>> from imod.prepare.topsystem import ALLOCATION_OPTION
    >>> setting = ALLOCATION_OPTION.at_first_active
    >>> allocated = allocate_rch_cells(setting, active, rate)
    """

    stage_to_riv_bot = 0
    first_active_to_elevation = -1
    stage_to_riv_bot_drn_above = 1
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
    """
    Allocate river cells from a planar grid across the vertical dimension.
    Multiple options are available, which can be selected in ALLOCATION_OPTION.

    Parameters
    ----------
    allocation_option: ALLOCATION_OPTION
        Chosen allocation option, can be selected using the ALLOCATION_OPTION
        enumerator.
    active: DataArray | UgridDatarray
        Boolean array containing active model cells. For Modflow 6, this is the
        equivalent of ``idomain == 1``.
    top: DataArray | UgridDatarray
        Grid containing tops of model layers. If has no layer dimension, is
        assumed as top of upper layer and the other layers are padded with
        bottom values of the overlying model layer.
    bottom: DataArray | UgridDatarray
        Grid containing bottoms of model layers.
    stage: DataArray | UgridDatarray
        Planar grid containing river stages. Is not allowed to have a layer
        dimension.
    bottom_elevation: DataArray | UgridDatarray
        Planar grid containing river bottom elevations. Is not allowed to have a
        layer dimension.

    Returns
    -------
    tuple(DataArray | UgridDatarray)
        Allocated river cells

    Examples
    --------

    >>> from imod.prepare.topsystem import ALLOCATION_OPTION, allocate_riv_cells
    >>> setting = ALLOCATION_OPTION.stage_to_riv_bot
    >>> allocated = allocate_riv_cells(setting, active, top, bottom, stage, bottom_elevation)
    """
    match allocation_option:
        case ALLOCATION_OPTION.stage_to_riv_bot:
            return _allocate_cells__stage_to_riv_bot(
                top, bottom, stage, bottom_elevation
            )
        case ALLOCATION_OPTION.first_active_to_elevation:
            return _allocate_cells__first_active_to_elevation(
                active, top, bottom, bottom_elevation
            )
        case ALLOCATION_OPTION.stage_to_riv_bot_drn_above:
            return _allocate_cells__stage_to_riv_bot_drn_above(
                active, top, bottom, stage, bottom_elevation
            )
        case ALLOCATION_OPTION.at_elevation:
            return _allocate_cells__at_elevation(top, bottom, bottom_elevation)
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active, bottom_elevation)
        case _:
            raise ValueError(
                "Received incompatible setting for rivers, only"
                f"'{ALLOCATION_OPTION.stage_to_riv_bot.name}' and"
                f"'{ALLOCATION_OPTION.first_active_to_elevation.name}' and"
                f"'{ALLOCATION_OPTION.stage_to_riv_bot_drn_above.name}' and"
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
    """
    Allocate drain cells from a planar grid across the vertical dimension.
    Multiple options are available, which can be selected in ALLOCATION_OPTION.

    Parameters
    ----------
    allocation_option: ALLOCATION_OPTION
        Chosen allocation option, can be selected using the ALLOCATION_OPTION
        enumerator.
    active: DataArray | UgridDatarray
        Boolean array containing active model cells. For Modflow 6, this is the
        equivalent of ``idomain == 1``.
    top: DataArray | UgridDatarray
        Grid containing tops of model layers. If has no layer dimension, is
        assumed as top of upper layer and the other layers are padded with
        bottom values of the overlying model layer.
    bottom: DataArray | UgridDatarray
        Grid containing bottoms of model layers.
    elevation: DataArray | UgridDatarray
        Planar grid containing drain elevation. Is not allowed to have a layer
        dimension.

    Returns
    -------
    DataArray | UgridDatarray
        Allocated drain cells

    Examples
    --------

    >>> from imod.prepare.topsystem import ALLOCATION_OPTION, allocate_drn_cells
    >>> setting = ALLOCATION_OPTION.at_elevation
    >>> allocated = allocate_drn_cells(setting, active, top, bottom, stage, drain_elevation)
    """
    match allocation_option:
        case ALLOCATION_OPTION.first_active_to_elevation:
            return _allocate_cells__first_active_to_elevation(
                active, top, bottom, elevation
            )[0]
        case ALLOCATION_OPTION.at_elevation:
            return _allocate_cells__at_elevation(top, bottom, elevation)[0]
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active, elevation)[0]
        case _:
            raise ValueError(
                "Received incompatible setting for drains, only"
                f"'{ALLOCATION_OPTION.first_active_to_elevation.name}', "
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
    """
    Allocate general head boundary (GHB) cells from a planar grid across the
    vertical dimension. Multiple options are available, which can be selected in
    ALLOCATION_OPTION.

    Parameters
    ----------
    allocation_option: ALLOCATION_OPTION
        Chosen allocation option, can be selected using the ALLOCATION_OPTION
        enumerator.
    active: DataArray | UgridDatarray
        Boolean array containing active model cells. For Modflow 6, this is the
        equivalent of ``idomain == 1``.
    top: DataArray | UgridDatarray
        Grid containing tops of model layers. If has no layer dimension, is
        assumed as top of upper layer and the other layers are padded with
        bottom values of the overlying model layer.
    bottom: DataArray | UgridDatarray
        Grid containing bottoms of model layers.
    head: DataArray | UgridDatarray
        Planar grid containing general head boundary's head. Is not allowed to
        have a layer dimension.

    Returns
    -------
    DataArray | UgridDatarray
        Allocated general head boundary cells

    Examples
    --------

    >>> from imod.prepare.topsystem import ALLOCATION_OPTION, allocate_ghb_cells
    >>> setting = ALLOCATION_OPTION.at_elevation
    >>> allocated = allocate_ghb_cells(setting, active, top, bottom, head)
    """
    match allocation_option:
        case ALLOCATION_OPTION.first_active_to_elevation:
            return _allocate_cells__first_active_to_elevation(
                active, top, bottom, head
            )[0]
        case ALLOCATION_OPTION.at_elevation:
            return _allocate_cells__at_elevation(top, bottom, head)[0]
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active, head)[0]
        case _:
            raise ValueError(
                "Received incompatible setting for general head boundary, only"
                f"'{ALLOCATION_OPTION.first_active_to_elevation.name}', "
                f"'{ALLOCATION_OPTION.at_elevation.name}' and"
                f"'{ALLOCATION_OPTION.at_first_active.name}' supported."
                f"got: '{allocation_option.name}'"
            )


def allocate_rch_cells(
    allocation_option: ALLOCATION_OPTION,
    active: GridDataArray,
    rate: GridDataArray,
) -> GridDataArray:
    """
    Allocate recharge cells from a planar grid across the vertical dimension.
    Multiple options are available, which can be selected in ALLOCATION_OPTION.

    Parameters
    ----------
    allocation_option: ALLOCATION_OPTION
        Chosen allocation option, can be selected using the ALLOCATION_OPTION
        enumerator.
    active: DataArray | UgridDataArray
        Boolean array containing active model cells. For Modflow 6, this is the
        equivalent of ``idomain == 1``.
    rate: DataArray | UgridDataArray
        Array with recharge rates. This will only be used to infer where
        recharge cells are defined.

    Returns
    -------
    DataArray | UgridDataArray
        Allocated recharge cells

    Examples
    --------

    >>> from imod.prepare.topsystem import ALLOCATION_OPTION, allocate_rch_cells
    >>> setting = ALLOCATION_OPTION.at_first_active
    >>> allocated = allocate_rch_cells(setting, active, rate)
    """
    match allocation_option:
        case ALLOCATION_OPTION.at_first_active:
            return _allocate_cells__at_first_active(active, rate)[0]
        case _:
            raise ValueError(
                "Received incompatible setting for recharge, only"
                f"'{ALLOCATION_OPTION.at_first_active.name}' supported."
                f"got: '{allocation_option.name}'"
            )


def _is_layered(grid: GridDataArray):
    return "layer" in grid.sizes and grid.sizes["layer"] > 1


def _enforce_layered_top(top: GridDataArray, bottom: GridDataArray):
    if _is_layered(top):
        return top
    else:
        return create_layered_top(bottom, top)


def get_above_lower_bound(bottom_elevation: GridDataArray, top_layered: GridDataArray):
    """
    Returns boolean array that indicates cells are above the lower vertical
    limit of the topsystem. These are the cells located above the
    bottom_elevation grid or in the first layer.
    """
    top_layer_label = {"layer": min(top_layered.coords["layer"])}
    is_above_lower_bound = bottom_elevation <= top_layered
    # Bottom elevation above top surface is allowed, so these are set to True
    # regardless.
    is_above_lower_bound.loc[top_layer_label] = ~bottom_elevation.isnull()
    return is_above_lower_bound


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
        river stage, cannot contain a layer dimension. Can contain a time
        dimension.
    bottom_elevation: GridDataArray
        river bottom elevation, cannot contain a layer dimension. Can contain a
        time dimension.

    Returns
    -------
    GridDataArray
        River cells
    """
    PLANAR_GRID.validate(stage)
    PLANAR_GRID.validate(bottom_elevation)

    top_layered = _enforce_layered_top(top, bottom)

    is_above_lower_bound = get_above_lower_bound(bottom_elevation, top_layered)
    is_below_upper_bound = stage > bottom
    riv_cells = is_below_upper_bound & is_above_lower_bound

    return riv_cells, None


@enforced_dim_order
def _allocate_cells__first_active_to_elevation(
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
        river bottom elevation, cannot contain a layer dimension. Can contain a
        time dimension.

    Returns
    -------
    GridDataArray
        River cells
    """
    PLANAR_GRID.validate(bottom_elevation)

    upper_active_layer = get_upper_active_layer_number(active)
    layer = active.coords["layer"]

    top_layered = _enforce_layered_top(top, bottom)

    is_above_lower_bound = get_above_lower_bound(bottom_elevation, top_layered)
    is_below_upper_bound = upper_active_layer <= layer
    riv_cells = is_below_upper_bound & is_above_lower_bound & active

    return riv_cells, None


@enforced_dim_order
def _allocate_cells__stage_to_riv_bot_drn_above(
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
        river stage, cannot contain a layer dimension. Can contain a time
        dimension.
    bottom_elevation: GridDataArray
        river bottom elevation, cannot contain a layer dimension. Can contain a
        time dimension.

    Returns
    -------
    riv_cells: GridDataArray
        River cells (below stage)
    drn_cells: GridDataArray
        Drainage cells (above stage)
    """

    PLANAR_GRID.validate(stage)
    PLANAR_GRID.validate(bottom_elevation)

    top_layered = _enforce_layered_top(top, bottom)
    is_above_lower_bound = get_above_lower_bound(bottom_elevation, top_layered)
    upper_active_layer = get_upper_active_layer_number(active)
    layer = active.coords["layer"]
    is_below_upper_bound = upper_active_layer <= layer
    is_below_upper_bound_and_active = is_below_upper_bound & active
    drn_cells = is_below_upper_bound_and_active & (bottom >= stage)
    riv_cells = (is_below_upper_bound_and_active & is_above_lower_bound) != drn_cells

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
        elevation. Can be river bottom, drain elevation or head of GHB. Cannot
        contain a layer dimension. Can contain a time dimension.

    Returns
    -------
    GridDataArray
        Topsystem cells
    """

    PLANAR_GRID.validate(elevation)

    top_layered = _enforce_layered_top(top, bottom)
    is_above_lower_bound = get_above_lower_bound(elevation, top_layered)
    is_below_upper_bound = elevation >= bottom
    riv_cells = is_below_upper_bound & is_above_lower_bound

    return riv_cells, None


@enforced_dim_order
def _allocate_cells__at_first_active(
    active: GridDataArray,
    planar_topsystem_grid: GridDataArray,
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
    planar_topsystem_grid: GridDataArray
        Grid with planar topsystem cells, assumed active where not nan.

    Returns
    -------
    GridDataArray
        Topsystem cells
    """
    PLANAR_GRID.validate(planar_topsystem_grid)

    upper_active = get_upper_active_grid_cells(active)
    topsystem_upper_active = upper_active & ~np.isnan(planar_topsystem_grid)

    return topsystem_upper_active, None
