import numbers
from dataclasses import dataclass

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
    return {
        key: _mask_spatial_var(grid, msw_active.all) for key, grid in cap_data.items()
    }


def mask_and_broadcast_pkg_data(
    package: type[MetaSwapPackage], grid_data: GridDataDict, msw_active: MetaSwapActive
) -> GridDataDict:
    """
    Mask and broadcast grid data, carefully mask per subunit if variable needs
    to contain subunits.
    """

    return {
        key: (
            _mask_spatial_var(grid, msw_active.per_subunit)
            if key in package._with_subunit
            else _mask_spatial_var(grid, msw_active.all)
        )
        for key, grid in grid_data.items()
    }


def _mask_spatial_var(da: GridDataArray, active: GridDataArray) -> GridDataArray:
    if issubclass(da.dtype.type, numbers.Integral):
        return da.where(active, other=0)
    elif issubclass(da.dtype.type, numbers.Real):
        return da.where(active)
    else:
        raise TypeError(
            f"Expected dtype float or integer. Received instead: {da.dtype}"
        )
