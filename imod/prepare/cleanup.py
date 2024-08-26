"""Cleanup utilities"""

from enum import Enum
from typing import Optional

import xarray as xr

from imod.mf6.utilities.mask import mask_arrays
from imod.schemata import scalar_None
from imod.typing import GridDataArray, GridDataset


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

    match method:
        case AlignLevelsMode.BOTTOMUP:
            return top.where(~to_align, bottom), bottom
        case AlignLevelsMode.TOPDOWN:
            return top, bottom.where(~to_align, top)
        case _:
            raise TypeError(f"Unmatched case for method, got {method}")


def _cleanup_robin_boundary(grids=dict[str, GridDataArray]) -> dict[str, GridDataArray]:
    """Cleanup robin boundary condition (i.e. bc with conductance)"""
    conductance = grids["conductance"]
    concentration = grids["concentration"]
    # Make conductance cells with erronous values inactive
    grids["conductance"] = conductance.where(conductance > 0.0)
    # Make concentration cells with erronous values inactive
    if (concentration is not None) and not scalar_None(concentration):
        grids["concentration"] = concentration.where(concentration >= 0.0)
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
    """
    Clean up river data, fixes some common mistakes causing ValidationErrors by
    doing the following:

    - Cells where conductance <= 0 are deactivated.
    - Cells where concentration < 0 are deactivated.
    - Align NoData: If one variable has an inactive cell in one cell, ensure
      this cell is deactivated for all variables.
    - River bottom elevations which exceed river stage are lowered to river
      stage.

    """
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
    """
    Clean up drain data, fixes some common mistakes causing ValidationErrors by
    doing the following:

    - Cells where conductance <= 0 are deactivated.
    - Cells where concentration < 0 are deactivated.
    - Align NoData: If one variable has an inactive cell in one cell, ensure
      this cell is deactivated for all variables.
    """
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
    """
    Clean up general head boundary data, fixes some common mistakes causing
    ValidationErrors by doing the following:

    - Cells where conductance <= 0 are deactivated.
    - Cells where concentration < 0 are deactivated.
    - Align NoData: If one variable has an inactive cell in one cell, ensure
      this cell is deactivated for all variables.
    """
    # Output dict
    output_dict = {
        "head": head,
        "conductance": conductance,
        "concentration": concentration,
    }
    return _cleanup_robin_boundary(output_dict)


def cleanup_wel(wel_ds: GridDataset):
    deactivate = wel_ds["screen_top"] < wel_ds["screen_bottom"]
    return wel_ds.where(~deactivate, drop=True)
