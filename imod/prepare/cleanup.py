"""Cleanup utilities"""

from enum import Enum
from typing import Optional

import xarray as xr

from imod.mf6.utilities.mask import mask_arrays
from imod.schemata import scalar_None
from imod.typing import GridDataArray


class AlignLevelsMode(Enum):
    TOPDOWN = 0
    BOTTOMUP = 1


def align_nodata(grids: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    return mask_arrays(grids)


def align_interface_levels(
    top: GridDataArray,
    bottom: GridDataArray,
    method: AlignLevelsMode = AlignLevelsMode.TOPDOWN,
) -> tuple[GridDataArray, GridDataArray]:
    to_align = top < bottom

    if method == AlignLevelsMode.BOTTOMUP:
        return top.where(~to_align, bottom), bottom
    elif method == AlignLevelsMode.TOPDOWN:
        return top, bottom.where(~to_align, top)
    else:
        raise TypeError("")


def _cleanup_robin_boundary(grids=dict[str, GridDataArray]) -> dict[str, GridDataArray]:
    """Cleanup robin boundary condition, with conductance"""
    conductance = grids["conductance"]
    concentration = grids["concentration"]
    # Make conductance cells with erronous values inactive
    grids["conductance"] = conductance.where(conductance <= 0.0)
    # Make concentration cells with erronous values inactive
    if (concentration is not None) or not scalar_None(concentration):
        grids["concentration"] = concentration.where(concentration < 0.0)
    else:
        grids.pop("concentration")

    # Align nodata
    return align_nodata(grids)


def cleanup_riv(
    stage: GridDataArray,
    conductance: GridDataArray,
    bottom_elevation: GridDataArray,
    concentration: Optional[GridDataArray] = None,
) -> dict[str, GridDataArray]:
    # Output dict
    output_dict = {
        "stage": stage,
        "conductance": conductance,
        "bottom_elevation": bottom_elevation,
        "concentration": concentration,
    }
    output_dict = _cleanup_robin_boundary(output_dict)
    # Ensure stage above bottom_elevation
    output_dict["stage"], output_dict["bottom_elevation"] = align_interface_levels(
        output_dict["stage"], output_dict["bottom_elevation"], AlignLevelsMode.TOPDOWN
    )
    return output_dict


def cleanup_drn(
    elevation: GridDataArray,
    conductance: GridDataArray,
    concentration: Optional[GridDataArray] = None,
) -> dict[str, GridDataArray]:
    # Output dict
    output_dict = {
        "elevation": elevation,
        "conductance": conductance,
        "concentration": concentration,
    }
    return _cleanup_robin_boundary(output_dict)


def cleanup_ghb(
    head: GridDataArray,
    conductance: GridDataArray,
    concentration: Optional[GridDataArray] = None,
) -> dict[str, GridDataArray]:
    # Output dict
    output_dict = {
        "head": head,
        "conductance": conductance,
        "concentration": concentration,
    }
    return _cleanup_robin_boundary(output_dict)
