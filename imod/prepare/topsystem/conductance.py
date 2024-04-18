from enum import Enum

import numpy as np

from imod.prepare.layer import (
    get_lower_active_grid_cells,
    get_upper_active_grid_cells,
)
from imod.prepare.topsystem.allocation import _enforce_layered_top
from imod.schemata import DimsSchema
from imod.typing import GridDataArray


class DISTRIBUTING_OPTION(Enum):
    """
    Enumerator for conductance distribution settings. Numbers match the
    DISTRCOND options in iMOD 5.6.
    """

    by_corrected_transmissivity = 0
    equally = 1
    by_crosscut_thickness = 2
    by_layer_thickness = 3
    by_crosscut_transmissivity = 4
    by_conductivity = 5
    by_layer_transmissivity = 9  # Not an iMOD 5.6 option


PLANAR_GRID = (
    DimsSchema("time", "y", "x")
    | DimsSchema("y", "x")
    | DimsSchema("time", "{face_dim}")
    | DimsSchema("{face_dim}")
)


def distribute_riv_conductance(
    distributing_option: DISTRIBUTING_OPTION,
    allocated: GridDataArray,
    conductance: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
    k: GridDataArray,
) -> GridDataArray:
    PLANAR_GRID.validate(conductance)

    match distributing_option:
        case DISTRIBUTING_OPTION.by_corrected_transmissivity:
            weights = _distribute_weights__by_corrected_transmissivity(
                allocated, top, bottom, stage, bottom_elevation, k
            )
        case DISTRIBUTING_OPTION.equally:
            weights = _distribute_weights__equally(allocated)
        case DISTRIBUTING_OPTION.by_crosscut_thickness:
            weights = _distribute_weights__by_crosscut_thickness(
                allocated, top, bottom, stage, bottom_elevation
            )
        case DISTRIBUTING_OPTION.by_layer_thickness:
            weights = _distribute_weights__by_layer_thickness(allocated, top, bottom)
        case DISTRIBUTING_OPTION.by_crosscut_transmissivity:
            weights = _distribute_weights__by_crosscut_transmissivity(
                allocated, top, bottom, stage, bottom_elevation, k
            )
        case DISTRIBUTING_OPTION.by_layer_transmissivity:
            weights = _distribute_weights__by_layer_transmissivity(
                allocated, top, bottom, k
            )
        case DISTRIBUTING_OPTION.by_conductivity:
            weights = _distribute_weights__by_conductivity(allocated, k)
        case _:
            raise ValueError(
                "Received incompatible setting for rivers, only"
                f"'{DISTRIBUTING_OPTION.by_corrected_transmissivity.name}' and"
                f"'{DISTRIBUTING_OPTION.equally.name}' and"
                f"'{DISTRIBUTING_OPTION.by_crosscut_thickness.name}' and"
                f"'{DISTRIBUTING_OPTION.by_layer_thickness.name}' and"
                f"'{DISTRIBUTING_OPTION.by_crosscut_transmissivity.name}' and"
                f"'{DISTRIBUTING_OPTION.by_layer_transmissivity.name}' and"
                f"'{DISTRIBUTING_OPTION.by_conductivity.name}' supported."
                f"got: '{distributing_option.name}'"
            )
    return (weights * conductance).where(allocated)


def distribute_drn_conductance(
    distributing_option: DISTRIBUTING_OPTION,
    allocated: GridDataArray,
    conductance: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    k: GridDataArray,
) -> GridDataArray:
    PLANAR_GRID.validate(conductance)

    match distributing_option:
        case DISTRIBUTING_OPTION.equally:
            weights = _distribute_weights__equally(allocated)
        case DISTRIBUTING_OPTION.by_layer_thickness:
            weights = _distribute_weights__by_layer_thickness(allocated, top, bottom)
        case DISTRIBUTING_OPTION.by_layer_transmissivity:
            weights = _distribute_weights__by_layer_transmissivity(
                allocated, top, bottom, k
            )
        case DISTRIBUTING_OPTION.by_conductivity:
            weights = _distribute_weights__by_conductivity(allocated, k)
        case _:
            raise ValueError(
                "Received incompatible setting for drains, only"
                f"'{DISTRIBUTING_OPTION.equally.name}' and"
                f"'{DISTRIBUTING_OPTION.by_layer_thickness.name}' and"
                f"'{DISTRIBUTING_OPTION.by_layer_transmissivity.name}' and"
                f"'{DISTRIBUTING_OPTION.by_conductivity.name}' supported."
                f"got: '{distributing_option.name}'"
            )
    return (weights * conductance).where(allocated)


def distribute_ghb_conductance(
    distributing_option: DISTRIBUTING_OPTION,
    allocated: GridDataArray,
    conductance: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    k: GridDataArray,
) -> GridDataArray:
    PLANAR_GRID.validate(conductance)

    match distributing_option:
        case DISTRIBUTING_OPTION.equally:
            weights = _distribute_weights__equally(allocated)
        case DISTRIBUTING_OPTION.by_layer_thickness:
            weights = _distribute_weights__by_layer_thickness(allocated, top, bottom)
        case DISTRIBUTING_OPTION.by_layer_transmissivity:
            weights = _distribute_weights__by_layer_transmissivity(
                allocated, top, bottom, k
            )
        case DISTRIBUTING_OPTION.by_conductivity:
            weights = _distribute_weights__by_conductivity(allocated, k)
        case _:
            raise ValueError(
                "Received incompatible setting for general head boundary, only"
                f"'{DISTRIBUTING_OPTION.equally.name}' and"
                f"'{DISTRIBUTING_OPTION.by_layer_thickness.name}' and"
                f"'{DISTRIBUTING_OPTION.by_layer_transmissivity.name}' and"
                f"'{DISTRIBUTING_OPTION.by_conductivity.name}' supported."
                f"got: '{distributing_option.name}'"
            )
    return (weights * conductance).where(allocated)


def _compute_layer_thickness(allocated, top, bottom):
    """
    Compute 3D grid of thicknesses in allocated cells
    """
    top_layered = _enforce_layered_top(top, bottom)

    thickness = top_layered - bottom
    return allocated * thickness


def _compute_crosscut_thickness(allocated, top, bottom, stage, bottom_elevation):
    """
    Compute 3D grid of thicknesses crosscut by river in allocated cells. So the
    upper allocated layer thickness is stage - bottom, the lower allocated layer
    is top - river_bottom_elevation.
    """
    top_layered = _enforce_layered_top(top, bottom)

    thickness = _compute_layer_thickness(allocated, top, bottom)

    upper_layer_allocated = get_upper_active_grid_cells(allocated)
    lower_layer_allocated = get_lower_active_grid_cells(allocated)

    thickness = thickness.where(
        ~upper_layer_allocated, thickness - (top_layered - stage)
    )
    thickness = thickness.where(
        ~lower_layer_allocated, thickness - (bottom_elevation - bottom)
    )

    return thickness


def _distribute_weights__by_corrected_transmissivity(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
    k: GridDataArray,
):
    PLANAR_GRID.validate(stage)
    PLANAR_GRID.validate(bottom_elevation)

    crosscut_thickness = _compute_crosscut_thickness(
        allocated, top, bottom, stage, bottom_elevation
    )
    transmissivity = crosscut_thickness * k

    top_layered = _enforce_layered_top(top, bottom)

    upper_layer_allocated = get_upper_active_grid_cells(allocated)
    lower_layer_allocated = get_lower_active_grid_cells(allocated)

    layer_thickness = _compute_layer_thickness(allocated, top, bottom)
    midpoints = (top_layered + bottom) / 2

    # Computing vertical midpoint of river crosscutting layers.
    Fc = midpoints.where(~upper_layer_allocated, (bottom + stage) / 2)
    Fc = Fc.where(~lower_layer_allocated, (top_layered + bottom_elevation) / 2)
    # Correction factor for mismatch between midpoints of crosscut layers and
    # layer midpoints.
    F = 1.0 - np.abs(midpoints - Fc) / (layer_thickness * 0.5)

    transmissivity_corrected = transmissivity * F
    return transmissivity_corrected / transmissivity_corrected.sum(dim="layer")


def _distribute_weights__equally(allocated: GridDataArray):
    weights = 1.0 / allocated.sum(dim="layer")
    return allocated * weights


def _distribute_weights__by_layer_thickness(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
):
    layer_thickness = _compute_layer_thickness(allocated, top, bottom)

    weights = layer_thickness / layer_thickness.sum(dim="layer")

    return weights


def _distribute_weights__by_crosscut_thickness(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
):
    PLANAR_GRID.validate(stage)
    PLANAR_GRID.validate(bottom_elevation)

    crosscut_thickness = _compute_crosscut_thickness(
        allocated, top, bottom, stage, bottom_elevation
    )

    return crosscut_thickness / (stage - bottom_elevation)


def _distribute_weights__by_layer_transmissivity(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    k: GridDataArray,
):

    layer_thickness = _compute_layer_thickness(
        allocated, top, bottom
    )
    transmissivity = layer_thickness * k

    return transmissivity / transmissivity.sum(dim="layer")


def _distribute_weights__by_crosscut_transmissivity(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
    k: GridDataArray,
):
    PLANAR_GRID.validate(stage)
    PLANAR_GRID.validate(bottom_elevation)

    crosscut_thickness = _compute_crosscut_thickness(
        allocated, top, bottom, stage, bottom_elevation
    )
    transmissivity = (crosscut_thickness * k)

    return transmissivity / transmissivity.sum(dim="layer")


def _distribute_weights__by_conductivity(allocated: GridDataArray, k: GridDataArray):
    k_allocated = allocated * k

    return k_allocated / k_allocated.sum(dim="layer")
