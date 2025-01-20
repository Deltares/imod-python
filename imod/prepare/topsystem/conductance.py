from enum import Enum
from typing import Optional

import numpy as np

from imod.prepare.topsystem.allocation import (
    _enforce_layered_top,
    get_above_lower_bound,
)
from imod.schemata import DimsSchema
from imod.typing import GridDataArray
from imod.typing.grid import ones_like, preserve_gridtype, zeros_like
from imod.util.dims import enforced_dim_order


class DISTRIBUTING_OPTION(Enum):
    """
    Enumerator containing settings to distribute 2D conductance grids over
    vertical layers for the RIV, DRN or GHB package.

    * ``by_corrected_transmissivity``: RIV. Distribute the conductance by
      corrected transmissivities. Crosscut thicknesses are used to compute
      transmissivities. The crosscut thicknesses is computed based on the
      overlap of bottom_elevation over the bottom allocated layer. Same holds
      for the stage and top allocated layer. Furthermore the method corrects
      distribution weights for the mismatch between the midpoints of crosscut
      areas and model layer midpoints. This is the default method in iMOD 5.6,
      thus DISTRCOND = 0.
    * ``equally``: RIV, DRN, GHB. Distribute conductances equally over layers.
      This matches iMOD 5.6 DISTRCOND = 1 option.
    * ``by_crosscut_thickness``: RIV. Distribute the conductance by crosscut
      thicknesses. The crosscut thicknesses is computed based on the overlap of
      bottom_elevation over the bottom allocated layer. Same holds for the stage
      and top allocated layer. This matches iMOD 5.6 DISTRCOND = 2 option.
    * ``by_layer_thickness``: RIV, DRN, GHB. Distribute the conductance by model
      layer thickness. This matches iMOD 5.6 DISTRCOND = 3 option.
    * ``by_crosscut_transmissivity``: RIV. Distribute the conductance by
      crosscut transmissivity. Crosscut thicknesses are used to compute
      transmissivities. The crosscut thicknesses is computed based on the
      overlap of bottom_elevation over the bottom allocated layer. Same holds
      for the stage and top allocated layer. This matches iMOD 5.6 DISTRCOND = 4
      option.
    * ``by_conductivity``: RIV, DRN, GHB. Distribute the conductance weighted by
      model layer hydraulic conductivities. This matches iMOD 5.6 DISTRCOND = 5
      option.
    * ``by_layer_transmissivity``: RIV, DRN, GHB. Distribute the conductance by
      model layer transmissivity. This has no equivalent in iMOD 5.6.
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
    k: GridDataArray,
    stage: GridDataArray,
    bottom_elevation: GridDataArray,
) -> GridDataArray:
    """
    Function to distribute 2D conductance over vertical layers for the RIV
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
    k: DataArray | UgridDataArray
        Hydraulic conductivities
    stage: DataArray | UgridDataArray
        Planar grid with river stages, cannot contain a layer dimension. Can
        contain a time dimension.
    bottom_elevation: DataArray | UgridDataArray
        Planar grid with river bottom elevations, cannot contain a layer
        dimension. Can contain a time dimension.

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
        case DISTRIBUTING_OPTION.by_crosscut_thickness:
            weights = _distribute_weights__by_crosscut_thickness(
                allocated, top, bottom, stage, bottom_elevation
            )
        case DISTRIBUTING_OPTION.by_crosscut_transmissivity:
            weights = _distribute_weights__by_crosscut_transmissivity(
                allocated, top, bottom, k, stage, bottom_elevation
            )
        case DISTRIBUTING_OPTION.by_corrected_transmissivity:
            weights = _distribute_weights__by_corrected_transmissivity(
                allocated, top, bottom, k, stage, bottom_elevation
            )
        case _:
            raise ValueError(
                "Received incompatible setting for rivers, only"
                f"'{DISTRIBUTING_OPTION.equally.name}', "
                f"'{DISTRIBUTING_OPTION.by_layer_thickness.name}', "
                f"'{DISTRIBUTING_OPTION.by_layer_transmissivity.name}', "
                f"'{DISTRIBUTING_OPTION.by_conductivity.name}', "
                f"'{DISTRIBUTING_OPTION.by_crosscut_thickness.name}', "
                f"'{DISTRIBUTING_OPTION.by_crosscut_transmissivity.name}', and "
                f"'{DISTRIBUTING_OPTION.by_corrected_transmissivity.name}' supported. "
                f"Got: '{distributing_option.name}'"
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
    elevation: GridDataArray,
) -> GridDataArray:
    """
    Function to distribute 2D conductance over vertical layers for the DRN
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
    elevation: DataArray | UgridDataArray
        Drain elevation

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
            conductance, top, bottom, k, drain_elevation
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
        case DISTRIBUTING_OPTION.by_crosscut_thickness:
            weights = _distribute_weights__by_crosscut_thickness(
                allocated, top, bottom, bc_bottom=elevation
            )
        case DISTRIBUTING_OPTION.by_crosscut_transmissivity:
            weights = _distribute_weights__by_crosscut_transmissivity(
                allocated, top, bottom, k, bc_bottom=elevation
            )
        case DISTRIBUTING_OPTION.by_corrected_transmissivity:
            weights = _distribute_weights__by_corrected_transmissivity(
                allocated, top, bottom, k, bc_bottom=elevation
            )
        case _:
            raise ValueError(
                "Received incompatible setting for drains, only"
                f"'{DISTRIBUTING_OPTION.equally.name}', "
                f"'{DISTRIBUTING_OPTION.by_layer_thickness.name}', "
                f"'{DISTRIBUTING_OPTION.by_layer_transmissivity.name}', "
                f"'{DISTRIBUTING_OPTION.by_conductivity.name}', "
                f"'{DISTRIBUTING_OPTION.by_crosscut_thickness.name}', "
                f"'{DISTRIBUTING_OPTION.by_crosscut_transmissivity.name}', and "
                f"'{DISTRIBUTING_OPTION.by_corrected_transmissivity.name}' supported. "
                f"Got: '{distributing_option.name}'"
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
    Function to distribute 2D conductance over vertical layers for the GHB
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
                f"'{DISTRIBUTING_OPTION.equally.name}', "
                f"'{DISTRIBUTING_OPTION.by_layer_thickness.name}', "
                f"'{DISTRIBUTING_OPTION.by_layer_transmissivity.name}', and "
                f"'{DISTRIBUTING_OPTION.by_conductivity.name}' supported. "
                f"Got: '{distributing_option.name}'"
            )
    return (weights * conductance).where(allocated)


@preserve_gridtype
def _compute_layer_thickness(
    allocated: GridDataArray, top: GridDataArray, bottom: GridDataArray
):
    """
    Compute 3D grid of thicknesses in allocated cells
    """
    top_layered = _enforce_layered_top(top, bottom)

    thickness = top_layered - bottom
    return thickness.where(allocated)


@preserve_gridtype
def _compute_crosscut_thickness(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    bc_top: Optional[GridDataArray] = None,
    bc_bottom: Optional[GridDataArray] = None,
):
    """
    Compute 3D grid of thicknesses crosscut by boundary condition (river/drain)
    in allocated cells. So the thickness in the upper allocated layer is bc_top
    - bottom and the lower allocated layer is top - bc_bottom.
    """
    if (bc_top is None) & (bc_bottom is None):
        raise ValueError("`bc_top` and `bc_bottom` cannot both be None.")

    top_layered = _enforce_layered_top(top, bottom)
    layer_thickness = _compute_layer_thickness(allocated, top, bottom)
    thickness = layer_thickness.copy()
    outside = zeros_like(allocated).astype(bool)

    if bc_top is not None:
        top_is_above_lower_bound = get_above_lower_bound(bc_top, top_layered)
        upper_layer_bc = top_is_above_lower_bound & (bc_top > bottom)
        outside = outside | (bc_top < bottom)
        thickness = thickness.where(~upper_layer_bc, thickness - (top_layered - bc_top))

    if bc_bottom is not None:
        bot_is_above_lower_bound = get_above_lower_bound(bc_bottom, top_layered)
        lower_layer_bc = bot_is_above_lower_bound & (bc_bottom > bottom)
        outside = outside | ~bot_is_above_lower_bound
        corrected_thickness = thickness - (bc_bottom - bottom)
        # Set top layer to 1.0, where top exceeds bc_bottom
        top_layer_label = {"layer": min(top_layered.coords["layer"])}
        is_below_surface = top_layered.loc[top_layer_label] > bc_bottom
        corrected_thickness.loc[top_layer_label] = corrected_thickness.loc[
            top_layer_label
        ].where(is_below_surface, 1.0)
        thickness = thickness.where(~lower_layer_bc, corrected_thickness)

    # Deal with case where bc_top and bc_bottom are equal.
    if (bc_top is not None) and (bc_bottom is not None):
        bc_top_equals_bc_bottom = bc_top == bc_bottom
        thickness = thickness.where(~bc_top_equals_bc_bottom, layer_thickness)

    thickness = thickness.where(~outside, 0.0)

    return thickness


def _distribute_weights__by_corrected_transmissivity(
    allocated: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    k: GridDataArray,
    bc_top: Optional[GridDataArray] = None,
    bc_bottom: Optional[GridDataArray] = None,
):
    """
    Distribute conductances according to default method in iMOD 5.6, as
    described page 690 of the iMOD 5.6 manual (but then to distribute WEL
    rates). The method uses crosscut thicknesses to compute transmissivities.
    Furthermore it corrects distribution weights for the mismatch between the
    midpoints of crosscut areas and layer midpoints.
    """
    crosscut_thickness = _compute_crosscut_thickness(
        allocated, top, bottom, bc_top=bc_top, bc_bottom=bc_bottom
    )
    transmissivity = crosscut_thickness * k

    top_layered = _enforce_layered_top(top, bottom)
    layer_thickness = _compute_layer_thickness(allocated, top, bottom)
    midpoints = (top_layered + bottom) / 2
    Fc = midpoints.copy()

    if bc_top is not None:
        PLANAR_GRID.validate(bc_top)
        top_is_above_lower_bound = get_above_lower_bound(bc_top, top_layered)
        upper_layer_bc = top_is_above_lower_bound & (bc_top > bottom)
        # Computing vertical midpoint of river crosscutting layers.
        Fc = Fc.where(~upper_layer_bc, (bottom + bc_top) / 2)

    if bc_bottom is not None:
        PLANAR_GRID.validate(bc_bottom)
        bot_is_above_lower_bound = get_above_lower_bound(bc_bottom, top_layered)
        lower_layer_bc = bot_is_above_lower_bound & (bc_bottom > bottom)
        # Computing vertical midpoint of river crosscutting layers.
        Fc = Fc.where(~lower_layer_bc, (top_layered + bc_bottom) / 2)

    # Correction factor for mismatch between midpoints of crosscut layers and
    # layer midpoints.
    F = 1.0 - np.abs(midpoints - Fc) / (layer_thickness * 0.5)
    # Negative values can be introduced when elevation above surface level, set
    # these to 1.0.
    top_layer_index = {"layer": min(top_layered.coords["layer"])}
    F_top_layer = F.loc[top_layer_index]
    F.loc[top_layer_index] = F_top_layer.where(F_top_layer > 0.0, 1.0)

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
    bc_top: Optional[GridDataArray] = None,
    bc_bottom: Optional[GridDataArray] = None,
):
    """Compute weights based on crosscut thickness"""
    if bc_top is not None:
        PLANAR_GRID.validate(bc_top)
    if bc_bottom is not None:
        PLANAR_GRID.validate(bc_bottom)

    crosscut_thickness = _compute_crosscut_thickness(
        allocated, top, bottom, bc_top, bc_bottom
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
    k: GridDataArray,
    bc_top: Optional[GridDataArray] = None,
    bc_bottom: Optional[GridDataArray] = None,
):
    """Compute weights based on crosscut transmissivity"""
    if bc_top is not None:
        PLANAR_GRID.validate(bc_top)
    if bc_bottom is not None:
        PLANAR_GRID.validate(bc_bottom)

    crosscut_thickness = _compute_crosscut_thickness(
        allocated, top, bottom, bc_top=bc_top, bc_bottom=bc_bottom
    )
    transmissivity = crosscut_thickness * k

    return transmissivity / transmissivity.sum(dim="layer")


def _distribute_weights__by_conductivity(allocated: GridDataArray, k: GridDataArray):
    """Compute weights based on hydraulic conductivity"""
    k_allocated = allocated * k

    return k_allocated / k_allocated.sum(dim="layer")


def split_conductance_with_infiltration_factor(
    conductance: GridDataArray, infiltration_factor: GridDataArray
) -> tuple[GridDataArray, GridDataArray]:
    """
    Seperates (exfiltration) conductance with an infiltration factor (iMODFLOW) into
    a drainage conductance and a river conductance following methods explained in Zaadnoordijk (2009).

    Parameters
    ----------
    conductance : xr.DataArray or float
        Exfiltration conductance. Is the default conductance provided to the iMODFLOW river package
    infiltration_factor : xr.DataArray or float
        Infiltration factor. The exfiltration conductance is multiplied with this factor to compute
        the infiltration conductance. If 0, no infiltration takes place; if 1, infiltration is equal to    exfiltration

    Returns
    -------
    drainage_conductance : xr.DataArray
        conductance for the drainage package
    river_conductance : xr.DataArray
        conductance for the river package

    Derivation
    ----------
    From Zaadnoordijk (2009):
    [1] cond_RIV = A/ci
    [2] cond_DRN = A * (ci-cd) / (ci*cd)
    Where cond_RIV and cond_DRN repsectively are the River and Drainage conductance [L^2/T],
    A is the cell area [L^2] and ci and cd respectively are the infiltration and exfiltration resistance [T]

    Taking f as the infiltration factor and cond_d as the exfiltration conductance, we can write (iMOD manual):
    [3] ci = cd * (1/f)
    [4] cond_d = A/cd

    We can then rewrite equations 1 and 2 to:
    [5] cond_RIV = f * cond_d
    [6] cond_DRN = (1-f) * cond_d

    References
    ----------
    Zaadnoordijk, W. (2009).
    Simulating Piecewise-Linear Surface Water and Ground Water Interactions with MODFLOW.
    Ground Water.
    https://ngwa.onlinelibrary.wiley.com/doi/10.1111/j.1745-6584.2009.00582.x

    iMOD manual v5.2 (2020)
    https://oss.deltares.nl/web/imod/

    """
    if np.any(infiltration_factor > 1):
        raise ValueError("The infiltration factor should not exceed 1")

    drainage_conductance = conductance * (1 - infiltration_factor)

    river_conductance = conductance * infiltration_factor

    # clean up the packages
    drainage_conductance = drainage_conductance.where(drainage_conductance > 0)
    river_conductance = river_conductance.where(river_conductance > 0)
    return drainage_conductance, river_conductance
