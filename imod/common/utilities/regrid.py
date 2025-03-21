import copy
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Optional, Union

import xarray as xr
from plum import Dispatcher
from xarray.core.utils import is_scalar
from xugrid.regrid.regridder import BaseRegridder

from imod.common.interfaces.ilinedatapackage import ILineDataPackage
from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.ipackage import IPackage
from imod.common.interfaces.ipointdatapackage import IPointDataPackage
from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.common.interfaces.isimulation import ISimulation
from imod.common.statusinfo import NestedStatusInfo
from imod.common.utilities.clip import clip_by_grid
from imod.common.utilities.regrid_method_type import EmptyRegridMethod, RegridMethodType
from imod.common.utilities.value_filters import is_valid
from imod.schemata import ValidationError
from imod.typing.grid import (
    GridDataArray,
    GridDataset,
    is_unstructured,
    ones_like,
)
from imod.util.regrid import (
    RegridderType,
    RegridderWeightsCache,
)

# create dispatcher instance to limit scope of typedispatching
dispatch = Dispatcher()


def handle_extra_coords(coordname: str, target_grid: GridDataArray, variable_data: Any):
    """
    If ``variable_data`` has a ``coords`` attribute and if ``coordname`` in
    ``target_grid``, copy coord. If ``coordname`` not in ``target_grid``, but in
    ``variable_data``, remove it.
    """
    if hasattr(variable_data, "coords"):
        if coordname in target_grid.coords:
            return variable_data.assign_coords(
                {coordname: target_grid.coords[coordname].values[()]}
            )
        elif coordname in variable_data.coords:
            return variable_data.drop_vars(coordname)

    return variable_data


def _regrid_array(
    da: GridDataArray,
    regridder_collection: RegridderWeightsCache,
    regridder_name: Union[RegridderType, BaseRegridder],
    regridder_function: Optional[str],
    target_grid: GridDataArray,
) -> Optional[GridDataArray]:
    """
    Regrids a GridDataArray. Each DataArray can represent:
    - a scalar value, valid for the whole grid
    - an array of a different scalar per layer
    - an array with a value per grid block
    - None
    """

    # skip regridding for scalar arrays with no valid values (such as "None")
    scalar_da: bool = is_scalar(da)
    if scalar_da and not is_valid(da.values[()]):
        return None

    # the dataarray might be a scalar. If it is, then it does not need regridding.
    if scalar_da:
        return da.values[()]

    if isinstance(da, xr.DataArray):
        coords = da.coords
        # if it is an xr.DataArray it may be layer-based; then no regridding is needed
        if not ("x" in coords and "y" in coords):
            return da

        # if it is an xr.DataArray it needs the dx, dy coordinates for regridding, which are otherwise not mandatory
        if not ("dx" in coords and "dy" in coords):
            raise ValueError(
                f"GridDataArray {da.name} does not have both a dx and dy coordinates"
            )

    # obtain an instance of a regridder for the chosen method
    regridder = regridder_collection.get_regridder(
        da,
        target_grid,
        regridder_name,
        regridder_function,
    )

    # store original dtype of data
    original_dtype = da.dtype

    # regrid data array
    regridded_array = regridder.regrid(da)

    # reconvert the result to the same dtype as the original
    return regridded_array.astype(original_dtype)


def _regrid_package_data(
    package_data: dict[str, GridDataArray] | GridDataset,
    target_grid: GridDataArray,
    regridder_settings: RegridMethodType,
    regrid_cache: RegridderWeightsCache,
    new_package_data: Optional[dict[str, GridDataArray]] = None,
) -> dict[str, GridDataArray]:
    """
    Regrid package data. Loops over regridder settings to regrid variables one
    by one. Variables not existent in the package data are skipped. Regridded
    package data is added to a dictionary, which can optionally be provided as
    argument to extend.
    """
    if new_package_data is None:
        new_package_data = {}

    settings_dict = RegridMethodType.asdict(regridder_settings)
    for (
        varname,
        regridder_type_and_function,
    ) in settings_dict.items():
        regridder_function: Optional[str] = None
        regridder_name = regridder_type_and_function[0]
        if len(regridder_type_and_function) > 1:
            regridder_function = regridder_type_and_function[1]

        # skip variables that are not in this dataset
        if varname not in package_data.keys():
            continue

        # regrid the variable
        new_package_data[varname] = _regrid_array(
            package_data[varname],
            regrid_cache,
            regridder_name,
            regridder_function,
            target_grid,
        )
        # set dx and dy if present in target_grid
        new_package_data[varname] = handle_extra_coords(
            "dx", target_grid, new_package_data[varname]
        )
        new_package_data[varname] = handle_extra_coords(
            "dy", target_grid, new_package_data[varname]
        )
    return new_package_data


def _get_unique_regridder_types(model: IModel) -> defaultdict[RegridderType, list[str]]:
    """
    This function loops over the packages and  collects all regridder-types that are in use.
    """
    methods: defaultdict = defaultdict(list)
    regrid_packages = [pkg for pkg in model.values() if isinstance(pkg, IRegridPackage)]
    regrid_packages_with_methods = {
        pkg: asdict(pkg.get_regrid_methods()).items()  # type: ignore # noqa: union-attr
        for pkg in regrid_packages
        if not isinstance(pkg.get_regrid_methods(), EmptyRegridMethod)
    }

    for pkg, regrid_methods in regrid_packages_with_methods.items():
        for variable, regrid_method in regrid_methods:
            if variable in pkg.dataset.data_vars:
                functiontype = None
                regriddertype = regrid_method[0]
                if len(regrid_method) > 1:
                    functiontype = regrid_method[1]
                if functiontype not in methods[regriddertype]:
                    methods[regriddertype].append(functiontype)
    return methods


@dispatch
def _regrid_like(
    package: IRegridPackage,
    target_grid: GridDataArray,
    regrid_cache: RegridderWeightsCache,
    regridder_types: Optional[RegridMethodType] = None,
    as_pkg_type: Optional[type[IRegridPackage]] = None,
) -> IPackage:
    """
    Creates a package of the same type as this package, based on another
    discretization. It regrids all the arrays in this package to the desired
    discretization, and leaves the options unmodified. At the moment only
    regridding to a different planar grid is supported, meaning ``target_grid``
    has different ``"x"`` and ``"y"`` or different ``cell2d`` coords.

    The default regridding methods are specified in the ``_regrid_method``
    attribute of the package. These defaults can be overridden using the
    input parameters of this function.

    Examples
    --------

    To regrid the npf package with a non-default method for the k-field, call
    regrid_like with these arguments:

    >>> regridder_types = imod.mf6.regrid.NodePropertyFlowRegridMethod(k=(imod.RegridderType.OVERLAP, "mean"))
    >>> new_npf = npf.regrid_like(like,  RegridderWeightsCache, regridder_types)

    Parameters
    ----------
    package: IRegridPackage:
        package to regrid
    target_grid: xr.DataArray or xu.UgridDataArray
        a grid defined using the same discretization as the one we want to regrid
        the package to
    regrid_cache: RegridderWeightsCache
        stores regridder weights for different regridders. Can be used to speed
        up regridding, if the same regridders are used several times for
        regridding different arrays.
    regridder_types: RegridMethodType, optional
        dictionary mapping arraynames (str) to a tuple of regrid type (a
        specialization class of BaseRegridder) and function name (str) this
        dictionary can be used to override the default mapping method.
    as_pkg_type: RegridPackageType, optional
        Package to initiate new package as. Is used to regrid
        StructuredDiscretization to VerticesDiscretization.

    Returns
    -------

    a package with the same options as this package, and with all the
    data-arrays regridded to another discretization, similar to the one used in
    input argument "target_grid"
    """
    if not hasattr(package, "_regrid_method"):
        raise NotImplementedError(
            f"Package {type(package).__name__} does not support regridding"
        )

    if as_pkg_type is None:
        as_pkg_type = package.__class__

    if regridder_types is None:
        regridder_types = package._regrid_method

    new_package_data = package.get_non_grid_data(regridder_types.asdict().keys())
    new_package_data = _regrid_package_data(
        package.dataset,
        target_grid,
        regridder_types,
        regrid_cache,
        new_package_data=new_package_data,
    )

    return as_pkg_type(**new_package_data)


@dispatch  # type: ignore[no-redef]
def _regrid_like(
    model: IModel,
    target_grid: GridDataArray,
    validate: bool = True,
    regrid_cache: Optional[RegridderWeightsCache] = None,
) -> IModel:
    """
    Creates a model by regridding the packages of this model to another
    discretization. It regrids all the arrays in the package using the default
    regridding methods. At the moment only regridding to a different planar grid
    is supported, meaning ``target_grid`` has different ``"x"`` and ``"y"`` or
    different ``cell2d`` coords.

    Parameters
    ----------
    target_grid: xr.DataArray or xu.UgridDataArray
        a grid defined using the same discretization as the one we want to
        regrid the package to
    validate: bool
        set to true to validate the regridded packages
    regrid_cache: RegridderWeightsCache, optional
        stores regridder weights for different regridders. Can be used to speed
        up regridding, if the same regridders are used several times for
        regridding different arrays.

    Returns
    -------

    a model with similar packages to the input model, and with all the
    data-arrays regridded to another discretization, similar to the one used in
    input argument "target_grid"
    """
    supported, error_with_object_name = model.is_regridding_supported()
    if not supported:
        raise ValueError(
            f"regridding this model cannot be done due to the presence of package {error_with_object_name}"
        )
    diskey = model._get_diskey()
    dis = model[diskey]
    if is_unstructured(dis["idomain"]) and not is_unstructured(target_grid):
        raise NotImplementedError(
            "Regridding unstructured model to a structured grid not supported."
        )

    new_model = model.__class__()
    if regrid_cache is None:
        regrid_cache = RegridderWeightsCache()
    for pkg_name, pkg in model.items():
        if isinstance(pkg, (IRegridPackage, ILineDataPackage, IPointDataPackage)):
            new_model[pkg_name] = pkg.regrid_like(target_grid, regrid_cache)
        else:
            raise NotImplementedError(
                f"regridding is not implemented for package {pkg_name} of type {type(pkg)}"
            )

    methods = _get_unique_regridder_types(model)
    output_domain = _get_regridding_domain(model, target_grid, regrid_cache, methods)
    output_domain = handle_extra_coords("dx", target_grid, output_domain)
    output_domain = handle_extra_coords("dy", target_grid, output_domain)
    new_model.mask_all_packages(output_domain)
    new_model.purge_empty_packages()
    if validate:
        status_info = NestedStatusInfo("Model validation status")
        status_info.add(new_model.validate("Regridded model"))
        if status_info.has_errors():
            raise ValidationError("\n" + status_info.to_string())
    return new_model


@dispatch  # type: ignore[no-redef]
def _regrid_like(
    simulation: ISimulation,
    regridded_simulation_name: str,
    target_grid: GridDataArray,
    validate: bool = True,
) -> ISimulation:
    """
    This method creates a new simulation object. The models contained in the new simulation are regridded versions
    of the models in the input object (this).
    Time discretization and solver settings are copied.

    Parameters
    ----------
    regridded_simulation_name: str
        name given to the output simulation
    target_grid: xr.DataArray or  xu.UgridDataArray
        discretization onto which the models  in this simulation will be regridded
    validate: bool
        set to true to validate the regridded packages

    Returns
    -------
    a new simulation object with regridded models
    """

    if simulation.is_split():
        raise RuntimeError(
            "Unable to regrid simulation. Regridding can only be done on simulations that haven't been split."
            + " Therefore regridding should be done before splitting the simulation."
        )
    if not simulation.has_one_flow_model():
        raise ValueError(
            "Unable to regrid simulation. Regridding can only be done on simulations that have a single flow model."
        )
    regrid_cache = RegridderWeightsCache()

    models = simulation.get_models()
    for model_name, model in models.items():
        supported, error_with_object_name = model.is_regridding_supported()
        if not supported:
            raise ValueError(
                f"Unable to regrid simulation, due to the presence of package '{error_with_object_name}' in model {model_name} "
            )

    result = simulation.__class__(regridded_simulation_name)
    for key, item in simulation.items():
        if isinstance(item, IModel):
            result[key] = item.regrid_like(target_grid, validate, regrid_cache)
        elif key == "gwtgwf_exchanges":
            pass
        elif isinstance(item, IPackage) and not isinstance(item, IRegridPackage):
            result[key] = copy.deepcopy(item)

        else:
            raise NotImplementedError(f"regridding not supported for {key}")

    return result


@dispatch  # type: ignore[no-redef]
def _regrid_like(
    package: ILineDataPackage, target_grid: GridDataArray, *_
) -> ILineDataPackage:
    """
    The regrid_like method is irrelevant for this package as it is
    grid-agnostic, instead this method clips the package based on the grid
    exterior.
    """
    return clip_by_grid(package, target_grid)


@dispatch  # type: ignore[no-redef]
def _regrid_like(
    package: IPointDataPackage, target_grid: GridDataArray, *_
) -> IPointDataPackage:
    """
    he regrid_like method is irrelevant for this package as it is
    grid-agnostic, instead this method clips the package based on the grid
    exterior.
    """
    target_grid_2d = target_grid.isel(layer=0, drop=True, missing_dims="ignore")
    return clip_by_grid(package, target_grid_2d)


@dispatch  # type: ignore[no-redef]
def _regrid_like(package: object, target_grid: GridDataArray, *_) -> None:
    raise TypeError("this object cannot be regridded")


def _get_regridding_domain(
    model: IModel,
    target_grid: GridDataArray,
    regrid_cache: RegridderWeightsCache,
    methods: defaultdict[RegridderType, list[str]],
) -> GridDataArray:
    """
    This method computes the output-domain for a regridding operation by regridding idomain with
    all regridders. Each regridder may leave some cells inactive. The output domain for the model consists of those
    cells that all regridders consider active.
    """
    idomain = model.domain
    included_in_all = ones_like(target_grid)
    regridders = [
        regrid_cache.get_regridder(idomain, target_grid, regriddertype, function)
        for regriddertype, functionlist in methods.items()
        for function in functionlist
    ]
    for regridder in regridders:
        regridded_idomain = regridder.regrid(idomain)
        included_in_all = included_in_all.where(regridded_idomain.notnull())
        included_in_all = regridded_idomain.where(
            regridded_idomain <= 0, other=included_in_all
        )

    new_idomain = included_in_all.where(included_in_all.notnull(), other=0)
    new_idomain = new_idomain.astype(int)

    return new_idomain
