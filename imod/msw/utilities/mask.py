from dataclasses import dataclass

from imod.common.constants import MaskValues
from imod.common.utilities.mask import mask_da
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.utilities.common import is_scalar_and_equal
from imod.typing import GridDataArray, GridDataDict, Imod5DataDict


@dataclass
class MetaSwapActive:
    all: GridDataArray
    per_subunit: GridDataArray


def mask_and_broadcast_cap_data(
    cap_data: GridDataDict,
    msw_active: MetaSwapActive,
    optional_inactive_variables: GridDataDict | None = None,
) -> GridDataDict:
    """
    Mask and broadcast cap data, always mask with "all" of MetaSwapActive.
    Optional inactive variables are preserved without masking or broadcasting.
    """
    if optional_inactive_variables is None:
        optional_inactive_variables = {}

    masked_data = {
        key: mask_da(grid, msw_active.all)
        for key, grid in cap_data.items()
        if key not in optional_inactive_variables
    }
    masked_data.update(optional_inactive_variables)
    return masked_data


def get_scalar_inactive_variables(
    imod5_cap: GridDataDict,
    optional_variables: list[str],
) -> GridDataDict:
    """
    Collect scalar inactive values for optional iMOD5 CAP variables.
    """
    return {
        variable: imod5_cap[variable]
        for variable in optional_variables
        if is_scalar_and_equal(imod5_cap[variable], MaskValues.msw_default)
    }


def mask_and_broadcast_imod5_data(
    imod5_data: Imod5DataDict,
    msw_active: MetaSwapActive,
    optional_variables: list[str],
) -> Imod5DataDict:
    """
    Mask and broadcast the CAP data of iMOD5 data. Optional variables that are
    scalar and inactive are checked before masking and preserved as scalars, as
    masking would broadcast them to grids.
    """
    cap_data = imod5_data["cap"]
    optional_inactive_variables = get_scalar_inactive_variables(
        cap_data, optional_variables
    )
    cap_data_masked = mask_and_broadcast_cap_data(
        cap_data,
        msw_active,
        optional_inactive_variables=optional_inactive_variables,
    )
    return {"cap": cap_data_masked, "extra": imod5_data["extra"]}


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
