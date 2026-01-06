import numbers

import xarray as xr
from plum import Dispatcher
from xarray.core.utils import is_scalar

from imod.common.interfaces.imaskingsettings import IMaskingSettings
from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.ipackage import IPackage
from imod.common.interfaces.isimulation import ISimulation
from imod.typing.grid import (
    GridDataArray,
    concat,
    get_spatial_dimension_names,
    is_same_domain,
    is_spatial_grid,
    notnull,
)

# create dispatcher instance to limit scope of typedispatching
dispatch = Dispatcher()


def _validate_coords_mask(mask: GridDataArray) -> None:
    """
    Validate that the coordinates of the mask are valid.
    """
    spatial_dimension_names = get_spatial_dimension_names(mask)
    # Add any additional dimensions that are not part of the spatial dimensions.
    # These are dimensions that are returned by xugrid
    additional_dimension_names = ["dx", "dy"]
    dimension_names = spatial_dimension_names + additional_dimension_names
    unexpected_coords = set(mask.coords) - set(dimension_names)
    if len(unexpected_coords) > 0:
        raise ValueError(
            f"Unexpected coordinates in masking domain: {unexpected_coords}"
        )


def mask_all_models(
    simulation: ISimulation,
    mask: GridDataArray,
    ignore_time_purge_empty: bool = False,
):
    _validate_coords_mask(mask)
    if simulation.is_split():
        raise ValueError(
            "masking can only be applied to simulations that have not been split. Apply masking before splitting."
        )

    flowmodels = list(simulation.get_models_of_type("gwf6").keys())
    transportmodels = list(simulation.get_models_of_type("gwt6").keys())

    modelnames = flowmodels + transportmodels

    for name in modelnames:
        if is_same_domain(simulation[name].domain, mask):
            simulation[name].mask_all_packages(mask, ignore_time_purge_empty)
        else:
            raise ValueError(
                "masking can only be applied to simulations when all the models in the simulation use the same grid."
            )


def mask_all_packages(
    model: IModel,
    mask: GridDataArray,
    ignore_time_purge_empty: bool = False,
):
    _validate_coords_mask(mask)
    for pkgname, pkg in model.items():
        model[pkgname] = pkg.mask(mask)
    model.purge_empty_packages(ignore_time=ignore_time_purge_empty)


def mask_package(package: IPackage, mask: GridDataArray) -> IPackage:
    masked = {}

    for var in package.dataset.data_vars.keys():
        if _skip_dataarray(package.dataset[var]) or _skip_variable(package, var):
            masked[var] = package.dataset[var]
        else:
            masked[var] = _mask_spatial_var(package, var, mask)

    return type(package)(**masked)


def _skip_dataarray(da: GridDataArray) -> bool:
    if len(da.dims) == 0 or set(da.dims).issubset(["layer", "time"]):
        return True

    if is_scalar(da):
        return True

    if not is_spatial_grid(da) and ("layer" not in da.dims):
        return True

    return False


@dispatch
def _skip_variable(package: IPackage, var: str) -> bool:
    return False


@dispatch  # type: ignore [no-redef]
def _skip_variable(package: IMaskingSettings, var: str) -> bool:
    return var in package.skip_variables


def _mask_spatial_var(self, var: str, mask: GridDataArray) -> GridDataArray:
    da = self.dataset[var]
    array_mask = _adjust_mask_for_unlayered_data(da, mask)
    active = array_mask > 0

    if issubclass(da.dtype.type, numbers.Integral):
        if var == "idomain":
            return da.where(active, other=array_mask)
        else:
            return da.where(active, other=0)
    elif issubclass(da.dtype.type, numbers.Real):
        return da.where(active)
    else:
        raise TypeError(
            f"Expected dtype float or integer. Received instead: {da.dtype}"
        )


def _adjust_mask_for_unlayered_data(
    da: GridDataArray, mask: GridDataArray
) -> GridDataArray:
    """
    Some arrays are not layered while the mask is layered (for example the
    top array in dis or disv packaged). In that case we use the top layer of
    the mask to perform the masking. If layer is not a dataset dimension,
    but still a dataset coordinate, we limit the mask to the relevant layer
    coordinate(s).
    """
    array_mask = mask
    if "layer" in da.coords and "layer" not in da.dims:
        array_mask = mask.sel(layer=da.coords["layer"])
    if "layer" not in da.coords and "layer" in array_mask.coords:
        array_mask = mask.isel(layer=0)

    return array_mask


def mask_arrays(arrays: dict[str, GridDataArray]) -> dict[str, GridDataArray]:
    """
    This function takes a dictionary of xr.DataArrays. The arrays are assumed to have the same
    coordinates. When a np.nan value is found in any array, the other arrays are also
    set to np.nan at the same coordinates.
    """
    masks = [notnull(array) for array in arrays.values()]
    # Get total mask across all arrays
    total_mask = concat(masks, dim="arrays").all("arrays")
    # Mask arrays with total mask
    arrays_masked = {key: array.where(total_mask) for key, array in arrays.items()}
    return arrays_masked


def broadcast_and_mask_arrays(
    arrays: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    """
    This function takes a dictionary of xr.DataArrays and broadcasts them to the same shape.
    It then masks the arrays with np.nan values where any of the arrays have np.nan values.
    """
    # Broadcast arrays to the same shape
    broadcasted_arrays = xr.broadcast(*arrays.values())
    # Test if there is a spatial grid in the arrays, if not the broadcasting
    # will result in no spatial grid.
    if not is_spatial_grid(broadcasted_arrays[0]):
        raise ValueError("One or more arrays need to be a spatial grid.")
    broadcasted_arrays = dict(zip(arrays.keys(), broadcasted_arrays))

    # Mask arrays with np.nan values
    return mask_arrays(broadcasted_arrays)
