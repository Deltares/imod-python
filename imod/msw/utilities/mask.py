from dataclasses import dataclass

from imod.common.utilities.mask import mask_da
from imod.msw.pkgbase import MetaSwapPackage
from imod.typing import GridDataArray, GridDataDict


@dataclass
class MetaSwapActive:
    all: GridDataArray
    per_subunit: GridDataArray


def mask_and_broadcast_cap_data(
    cap_data: GridDataDict, msw_active: MetaSwapActive
) -> GridDataDict:
    """
    Mask and broadcast cap data, always mask with "all" of MetaSwapActive.
    """
    return {key: mask_da(grid, msw_active.all) for key, grid in cap_data.items()}


def mask_and_broadcast_pkg_data(
    package: type[MetaSwapPackage], grid_data: GridDataDict, msw_active: MetaSwapActive
) -> GridDataDict:
    """
    Mask and broadcast grid data, carefully mask per subunit if variable needs
    to contain subunits.
    """

    return {
        key: (
            mask_da(grid, msw_active.per_subunit)
            if key in package._with_subunit
            else mask_da(grid, msw_active.all)
        )
        for key, grid in grid_data.items()
    }
