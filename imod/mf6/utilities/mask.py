
import numbers

import numpy as np
from xarray.core.utils import is_scalar

from imod.mf6.auxiliary_variables import (
        expand_transient_auxiliary_variables,
        remove_expanded_auxiliary_variables_from_dataset,
)
from imod.mf6.interfaces.imodel import IModel
from imod.mf6.interfaces.ipackage import IPackage
from imod.mf6.interfaces.isimulation import ISimulation
from imod.typing.grid import GridDataArray, get_spatial_dimension_names, is_same_domain


def _mask_all_models(
        simulation: ISimulation,
        mask: GridDataArray,
    ):
        spatial_dims = get_spatial_dimension_names(mask)
        if any([coord not in spatial_dims for coord in mask.coords]):
            raise ValueError("unexpected coordinate dimension in masking domain")
        

        if simulation.is_split():
            raise ValueError("masking can only be applied to simulations that have not been split. Apply masking before splitting.")                    

        flowmodels =list(simulation.get_models_of_type("gwf6").keys())
        transportmodels = list(simulation.get_models_of_type("gwt6").keys())      
        modelnames = flowmodels + transportmodels


        for name in modelnames:
            if is_same_domain(simulation[name].domain, mask):
                simulation[name].mask_all_packages(mask)
            else:
                raise ValueError("masking can only be applied to simulations when all the models in the simulation use the same grid.")
            

def _mask_all_packages(
    model: IModel,
    mask: GridDataArray,
):
    if any([coord not in ["x", "y", "layer", "mesh2d_nFaces", "dx", "dy"] for coord in mask.coords]):
        raise ValueError("unexpected coordinate dimension in masking domain")

    for pkgname, pkg in model.items():
        model[pkgname] = pkg.mask(mask)
    model.purge_empty_packages()


def _mask(package: IPackage,  mask: GridDataArray) -> IPackage:
    masked = {}
    if len(package.auxiliary_data_fields) > 0:
        remove_expanded_auxiliary_variables_from_dataset(package)
    for var in package.dataset.data_vars.keys():
        if _skip_masking_variable(package, var, package.dataset[var]):
            masked[var] = package.dataset[var]
        else:
            masked[var] = _mask_spatial_var(package, var, mask)
    if len(package.auxiliary_data_fields) > 0:
        expand_transient_auxiliary_variables(package)
    return type(package)(**masked)


def _skip_masking_variable(package: IPackage, var: str, da: GridDataArray)->bool: 
    if package._skip_masking_dataarray(var) or len(da.dims) == 0 or set(da.coords).issubset(["layer"]):
        return True
    if is_scalar(da.values[()]):
        return True
    spatial_dims = ["x", "y", "mesh2d_nFaces", "layer"]
    if not np.any( [coord in spatial_dims for coord in da.coords]):
        return True
    return False




def _mask_spatial_var(self, var: str, mask: GridDataArray)->GridDataArray:
    da = self.dataset[var]
    array_mask = _adjust_mask_for_unlayered_data(da, mask)

    if issubclass(da.dtype.type, numbers.Integral):
        if var == "idomain":
            return da.where(array_mask > 0, other=array_mask)
        else:
            return da.where(array_mask > 0, other=0)
    elif issubclass(da.dtype.type, numbers.Real):
        return da.where(array_mask > 0)
    else:
        raise TypeError(
            f"Expected dtype float or integer. Received instead: {da.dtype}"
        )

def _adjust_mask_for_unlayered_data(da: GridDataArray, mask: GridDataArray)->GridDataArray:
    '''
    Some arrays are not layered while the mask is layered (for example the
    top array in dis or disv packaged). In that case we use the top layer of
    the mask to perform the masking. If layer is not a dataset dimension,
    but still a dataset coordinate, we limit the mask to the relevant layer
    coordinate(s). 
    '''
    array_mask  = mask
    if "layer" in da.coords and "layer" not in da.dims:
        array_mask = mask.sel(layer=da.coords["layer"])        
    if "layer" not in da.coords and "layer" in array_mask.coords:
        array_mask = mask.isel(layer=0)

    return array_mask         