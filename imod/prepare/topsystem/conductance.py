from enum import Enum

import numpy as np

from imod.prepare.topsystem.allocation import _enforce_layered_top
from imod.schemata import DimsSchema
from imod.typing import GridDataArray
from imod.typing.grid import ones_like, preserve_gridtype
from imod.util.dims import enforced_dim_order


class DISTRIBUTING_OPTION(Enum):
    """
    Enumerator for conductance distribution settings. Numbers match the
    DISTRCOND options in iMOD 5.6. The following settings are available:

    * ``by_corrected_transmissivity``: Distribute the conductance by corrected
      transmissivities. Crosscut thicknesses are used to compute
      transmissivities. The crosscut thicknesses is computed based on the
      overlap of bottom_elevation over the bottom allocated layer. Same holds
      for the stage and top allocated layer. Furthermore the method corrects
      distribution weights for the mismatch between the midpoints of crosscut
      areas and model layer midpoints. This is the default method in iMOD 5.6,
      thus DISTRCOND = 0.
    * ``equally``: Distribute conductances equally over layers. This matches iMOD
      5.6 DISTRCOND = 1 option.
    * ``by_crosscut_thickness``: Distribute the conductance by crosscut
      thicknesses. The crosscut thicknesses is computed based on the overlap of
      bottom_elevation over the bottom allocated layer. Same holds for the stage
      and top allocated layer. This matches iMOD 5.6 DISTRCOND = 2 option.
    * ``by_layer_thickness``: Distribute the conductance by model layer thickness.
      This matches iMOD 5.6 DISTRCOND = 3 option.
    * ``by_crosscut_transmissivity``: Distribute the conductance by crosscut
      transmissivity. Crosscut thicknesses are used to compute transmissivities.
      The crosscut thicknesses is computed based on the overlap of
      bottom_elevation over the bottom allocated layer. Same holds for the stage
      and top allocated layer. This matches iMOD 5.6 DISTRCOND = 4 option.
    * ``by_conductivity``: Distribute the conductance weighted by model layer
      hydraulic conductivities. This matches iMOD 5.6 DISTRCOND = 5 option.
    * ``by_layer_transmissivity``: Distribute the conductance by model layer
      transmissivity. This has no equivalent in iMOD 5.6.
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


@enforced_dim_order
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
    """
    Function to distribute 2D conductance over vertical layer for the RIV
    package. Multiple options are available, which need to be selected in the
    DISTRIBUTING_OPTION enumerator.

    Parameters
    ----------
    distributing_option : DISTRIBUTING_OPTION
        Distributing option available in the DISTRIBUTING_OPTION enumerator.
    allocated: DataArray | UgridDataArray
        3D boolean array with river cell locations. This can be made with the
        :func:`imod.prepare.allocate_riv_cells` function.
    conductance: DataArray | UgridDataArray
        Planar grid with conductances that need to be distributed over layers,
        so grid cannot contain a layer dimension. Can contain a time dimension.
    top: DataArray | UgridDataArray
        Model top
    bottom: DataArray | UgridDataArray
        Model layer bottoms
    stage: DataArray | UgridDataArray
        Planar grid with river stages, cannot contain a layer dimension. Can
        contain a time dimension.
    bottom_elevation: DataArray | UgridDataArray
        Planar grid with river bottom elevations, cannot contain a layer
        dimension. Can contain a time dimension.
    k: DataArray | UgridDataArray
        Hydraulic conductivities

    Returns
    -------
    Conductances distributed over depth.

    Examples
    --------
    >>> from imod.prepare import allocate_riv_cells, distribute_riv_conductance, ALLOCATION_OPTION, DISTRIBUTING_OPTION
    >>> allocated = allocate_riv_cells(
        ALLOCATION_OPTION.stage_to_riv_bot, active, top, bottom, stage, bottom_elevation
        )
    >>> conductances_distributed = distribute_riv_conductance(
            DISTRIBUTING_OPTION.by_corrected_transmissivity, allocated,
            conductance, top, bottom, stage, bottom_elevation, k
        )
    """
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


@enforced_dim_order
def distribute_drn_conductance(
    distributing_option: DISTRIBUTING_OPTION,
    allocated: GridDataArray,
    conductance: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    k: GridDataArray,
) -> GridDataArray:
    """
    Function to distribute 2D conductance over vertical layer for the DRN
    package. Multiple options are available, which need to be selected in the
    DISTRIBUTING_OPTION enumerator.

    Parameters
    ----------
    distributing_option : DISTRIBUTING_OPTION
        Distributing option available in the DISTRIBUTING_OPTION enumerator.
    allocated: DataArray | UgridDataArray
        3D boolean array with drain cell locations. This can be made with the
        :func:`imod.prepare.allocate_drn_cells` function.
    conductance: DataArray | UgridDataArray
        Planar grid with conductances that need to be distributed over layers,
        so grid cannot contain a layer dimension. Can contain a time dimension.
    top: DataArray | UgridDataArray
        Model top
    bottom: DataArray | UgridDataArray
        Model layer bottoms
    k: DataArray | UgridDataArray
        Hydraulic conductivities

    Returns
    -------
    Conductances distributed over depth.

    Examples
    --------
    >>> from imod.prepare import allocate_drn_cells, distribute_drn_conductance, ALLOCATION_OPTION, DISTRIBUTING_OPTION
    >>> allocated = allocate_drn_cells(
        ALLOCATION_OPTION.at_elevation, active, top, bottom, drain_elevation
        )
    >>> conductances_distributed = distribute_drn_conductance(
            DISTRIBUTING_OPTION.by_layer_transmissivity, allocated,
            conductance, top, bottom, k
        )
    """
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


@enforced_dim_order
def distribute_ghb_conductance(
    distributing_option: DISTRIBUTING_OPTION,
    allocated: GridDataArray,
    conductance: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    k: GridDataArray,
) -> GridDataArray:
    PLANAR_GRID.validate(conductance)
    """
    Function to distribute 2D conductance over vertical layer for the GHB
    package. Multiple options are available, which need to be selected in the
    DISTRIBUTING_OPTION enumerator.

    Parameters
    ----------
    distributing_option : DISTRIBUTING_OPTION
        Distributing option available in the DISTRIBUTING_OPTION enumerator.
    allocated: DataArray | UgridDataArray
        3D boolean array with GHB cell locations. This can be made with the
        :func:`imod.prepare.allocate_ghb_cells` function.
    conductance: DataArray | UgridDataArray
        Planar grid with conductances that need to be distributed over layers,
        so grid cannot contain a layer dimension. Can contain a time dimension.
    top: DataArray | UgridDataArray
        Model top
    bottom: DataArray | UgridDataArray
        Model layer bottoms
    k: DataArray | UgridDataArray
        Hydraulic conductivities
    
    Returns
    -------
    Conductances distributed over depth.

    Examples
    --------
    >>> from imod.prepare import allocate_ghb_cells, distribute_drn_conductance, ALLOCATION_OPTION, DISTRIBUTING_OPTION
    >>> allocated = allocate_ghb_cells(
        ALLOCATION_OPTION.at_elevation, active, top, bottom, ghb_head
        )
    >>> conductances_distributed = distribute_ghb_conductance(
            DISTRIBUTING_OPTION.by_layer_transmissivity, allocated, 
            conductance, top, bottom, k
        )
    """
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


@preserve_gridtype
def _compute_layer_thickness(allocated, top, bottom):
    """
    Compute 3D grid of thicknesses in allocated cells
    """
    top_layered = _enforce_layered_top(top, bottom)

    thickness = top_layered - bottom
    return thickness.where(allocated)


@preserve_gridtype
def _compute_crosscut_thickness(allocated, top, bottom, stage, bottom_elevation):
    """
    Compute 3D grid of thicknesses crosscut by river in allocated cells. So the
    upper allocated layer thickness is stage - bottom, the lower allocated layer
    is top - river_bottom_elevation.
    """
    top_layered = _enforce_layered_top(top, bottom)

    thickness = _compute_layer_thickness(allocated, top, bottom)

    upper_layer_allocated = (stage < top_layered) & (stage > bottom)
    lower_layer_allocated = (bottom_elevation < top_layered) & (bottom_elevation > bottom)
    outside = (stage < bottom) | (bottom_elevation > top_layered)

    thickness = thickness.where(
        ~upper_layer_allocated, thickness - (top_layered - stage)
    )
    thickness = thickness.where(
        ~lower_layer_allocated, thickness - (bottom_elevation - bottom)
    )
    thickness = thickness.where(~outside, 0.0)

    return thickness

def _distribute_weights__by_corrected_transmissivity(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
    k: GridDataArray,
):
    """
    Distribute conductances according to default method in iMOD 5.6, as
    described page 690 of the iMOD 5.6 manual (but then to distribute WEL
    rates). The method uses crosscut thicknesses to compute transmissivities.
    Furthermore it corrects distribution weights for the mismatch between the
    midpoints of crosscut areas and layer midpoints.
    """
    PLANAR_GRID.validate(stage)
    PLANAR_GRID.validate(bottom_elevation)

    crosscut_thickness = _compute_crosscut_thickness(
        allocated, top, bottom, stage, bottom_elevation
    )
    transmissivity = crosscut_thickness * k

    top_layered = _enforce_layered_top(top, bottom)

    upper_layer_allocated = (stage < top_layered) & (stage > bottom)
    lower_layer_allocated = (bottom_elevation < top_layered) & (bottom_elevation > bottom)

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
    """Compute weights over layers equally."""
    return ones_like(allocated) / allocated.sum(dim="layer")


def _distribute_weights__by_layer_thickness(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
):
    """Compute weights based on layer thickness"""
    layer_thickness = _compute_layer_thickness(allocated, top, bottom)

    return layer_thickness / layer_thickness.sum(dim="layer")


def _distribute_weights__by_crosscut_thickness(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
):
    """Compute weights based on crosscut thickness"""
    PLANAR_GRID.validate(stage)
    PLANAR_GRID.validate(bottom_elevation)

    crosscut_thickness = _compute_crosscut_thickness(
        allocated, top, bottom, stage, bottom_elevation
    ).where(allocated)

    return crosscut_thickness / crosscut_thickness.sum(dim="layer")


def _distribute_weights__by_layer_transmissivity(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    k: GridDataArray,
):
    """Compute weights based on layer transmissivity"""
    layer_thickness = _compute_layer_thickness(allocated, top, bottom)
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
    """Compute weights based on crosscut transmissivity"""
    PLANAR_GRID.validate(stage)
    PLANAR_GRID.validate(bottom_elevation)

    crosscut_thickness = _compute_crosscut_thickness(
        allocated, top, bottom, stage, bottom_elevation
    )
    transmissivity = crosscut_thickness * k

    return transmissivity / transmissivity.sum(dim="layer")


def _distribute_weights__by_conductivity(allocated: GridDataArray, k: GridDataArray):
    """Compute weights based on hydraulic conductivity"""
    k_allocated = allocated * k

    return k_allocated / k_allocated.sum(dim="layer")
