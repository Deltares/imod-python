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
    Enumerator for settings to allocate planar grid with RIV, DRN, GHB, or RCH
    cells over the vertical layer dimensions. Numbers match the IDEFLAYER
    options in iMOD 5.6.

    * ``stage_to_riv_bot``: RIV. Allocate cells spanning from the river stage up
      to the river bottom elevation. This matches the iMOD 5.6 IDEFLAYER = 0
      option.
    * ``first_active_to_riv_bot``: RIV. Allocate cells spanning from first upper
      active cell up to the river bottom elevation. This matches the iMOD 5.6
      IDEFLAYER = -1 option.
    * ``first_active_to_riv_bot__drn``: RIV. Allocate cells spanning from first
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
    >>> allocated = allocate_rch_cells(setting, active)
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
    DataArray | UgridDatarray
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
    """
    Allocate recharge cells from a planar grid across the vertical dimension.
    Multiple options are available, which can be selected in ALLOCATION_OPTION.

    Parameters
    ----------
    allocation_option: ALLOCATION_OPTION
        Chosen allocation option, can be selected using the ALLOCATION_OPTION
        enumerator.
    active: DataArray | UgridDatarray
        Boolean array containing active model cells. For Modflow 6, this is the
        equivalent of ``idomain == 1``.

    Returns
    -------
    DataArray | UgridDatarray
        Allocated recharge cells

    Examples
    --------

    >>> from imod.prepare.topsystem import ALLOCATION_OPTION, allocate_rch_cells
    >>> setting = ALLOCATION_OPTION.at_first_active
    >>> allocated = allocate_rch_cells(setting, active)
    """
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


def _enforce_layered_top(top: GridDataArray, bottom: GridDataArray):
    if _is_layered(top):
        return top
    else:
        return create_layered_top(bottom, top)


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
        elevation. Can be river bottom, drain elevation or head of GHB. Cannot
        contain a layer dimension. Can contain a time dimension.

    Returns
    -------
    GridDataArray
        River cells
    """

    PLANAR_GRID.validate(elevation)

    top_layered = _enforce_layered_top(top, bottom)

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
