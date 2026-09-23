from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.itopsystembc import ITopSystemBoundaryCondition
from imod.typing import GridDataArray


def mask_topsystem(model: IModel, is_active: GridDataArray):
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
    """
    topsystem_packages = [
        key
        for key, pkg in model.items()
        if isinstance(pkg, ITopSystemBoundaryCondition)
    ]
    for key in topsystem_packages:
        model[key] = model[key].mask(is_active)
