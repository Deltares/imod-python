import numpy as np

from imod.common.interfaces.iboundarycondition import IBoundaryCondition
from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.itopsystembc import ITopSystemBoundaryCondition
from imod.typing import GridDataArray


def mask_topsystem(
    model: IModel, is_active: GridDataArray, ignore_time_purge_empty: bool
) -> None:
    """
    Mask all top system packages in the model inplace with a boolean mask
    indicating active cells.

    Parameters
    ----------
    model : IModel
        The MODFLOW 6 model containing top system packages.
    is_active : GridDataArray
        A boolean array indicating active cells. Top system packages will be masked
        where this array is False.
    ignore_time_purge_empty : bool
        If True, ignore the time dimension when masking the packages.
    """
    topsystem_packages = [
        key
        for key, pkg in model.items()
        if isinstance(pkg, ITopSystemBoundaryCondition)
    ]
    model.mask_packages(topsystem_packages, is_active, ignore_time_purge_empty)


def mask_topsystem_where_bc(
    model: IModel, boundary_condition: IBoundaryCondition, ignore_time: bool
) -> None:
    """
    Mask all top system packages in the model inplace where the boundary
    condition is active.

    Parameters
    ----------
    model : IModel
        The MODFLOW 6 model containing top system packages.
    boundary_condition : IBoundaryCondition
        The boundary condition package used to determine which cells are not
        added.
    ignore_time : bool
        If True, ignore the time dimension when masking the packages and set the
        mask to where the first time step of the boundary condition contains
        active cells. Else, aggregate over all time steps to determine the mask.
    """
    state_varname = boundary_condition._period_data[0]
    state_var = boundary_condition.dataset[state_varname]
    if "time" in state_var.dims:
        if ignore_time:
            state_var = state_var.isel(time=0, drop=True)
        else:
            state_var = state_var.min(dim="time")
    not_added_bc = np.isnan(state_var)
    mask_topsystem(model, not_added_bc, ignore_time)
