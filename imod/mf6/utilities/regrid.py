import abc
import copy
from collections import defaultdict
from typing import Any, Optional, Union

import xarray as xr
import xugrid as xu
from fastcore.dispatch import typedispatch
from xarray.core.utils import is_scalar
from xugrid.regrid.regridder import BaseRegridder

from imod.mf6.auxiliary_variables import (
    expand_transient_auxiliary_variables,
    remove_expanded_auxiliary_variables_from_dataset,
)
from imod.mf6.interfaces.ilinedatapackage import ILineDataPackage
from imod.mf6.interfaces.imodel import IModel
from imod.mf6.interfaces.ipackage import IPackage
from imod.mf6.interfaces.ipointdatapackage import IPointDataPackage
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.interfaces.isimulation import ISimulation
from imod.mf6.statusinfo import NestedStatusInfo
from imod.mf6.utilities.clip import clip_by_grid
from imod.mf6.utilities.regridding_types import RegridderType
from imod.schemata import ValidationError
from imod.typing.grid import GridDataArray, ones_like


class RegridderInstancesCollection:
    """
    This class stores any number of regridders that can regrid a single source grid to a single target grid.
    By storing the regridders, we make sure the regridders can be re-used for different arrays on the same grid.
    This is important because computing the regridding weights is a costly affair.
    """

    def __init__(
        self,
        source_grid: Union[xr.DataArray, xu.UgridDataArray],
        target_grid: Union[xr.DataArray, xu.UgridDataArray],
    ) -> None:
        self.regridder_instances: dict[
            tuple[type[BaseRegridder], Optional[str]], BaseRegridder
        ] = {}
        self._source_grid = source_grid
        self._target_grid = target_grid

    def __has_regridder(
        self, regridder_type: type[BaseRegridder], method: Optional[str] = None
    ) -> bool:
        return (regridder_type, method) in self.regridder_instances.keys()

    def __get_existing_regridder(
        self, regridder_type: type[BaseRegridder], method: Optional[str]
    ) -> BaseRegridder:
        if self.__has_regridder(regridder_type, method):
            return self.regridder_instances[(regridder_type, method)]
        raise ValueError("no existing regridder of type " + str(regridder_type))

    def __create_regridder(
        self, regridder_type: type[BaseRegridder], method: Optional[str]
    ) -> BaseRegridder:
        method_args = () if method is None else (method,)

        self.regridder_instances[(regridder_type, method)] = regridder_type(
            self._source_grid, self._target_grid, *method_args
        )
        return self.regridder_instances[(regridder_type, method)]

    def __get_regridder_class(
        self, regridder_type: RegridderType | BaseRegridder
    ) -> type[BaseRegridder]:
        if isinstance(regridder_type, abc.ABCMeta):
            if not issubclass(regridder_type, BaseRegridder):
                raise ValueError(
                    "only derived types of BaseRegridder can be instantiated"
                )
            return regridder_type
        elif isinstance(regridder_type, RegridderType):
            return regridder_type.value

        raise ValueError("invalid type for regridder")

    def get_regridder(
        self,
        regridder_type: Union[RegridderType, BaseRegridder],
        method: Optional[str] = None,
    ) -> BaseRegridder:
        """
        returns a regridder of the specified type and with the specified method.
        The desired type can be passed through  the argument "regridder_type" as an enumerator or
        as a class.
        The following two are equivalent:
        instancesCollection.get_regridder(RegridderType.OVERLAP, "mean")
        instancesCollection.get_regridder(xu.OverlapRegridder, "mean")


        Parameters
        ----------
        regridder_type: RegridderType or regridder class
            indicates the desired regridder type
        method: str or None
            indicates the method the regridder should apply

        Returns
        -------
        a regridder of the specified characteristics
        """
        regridder_class = self.__get_regridder_class(regridder_type)

        if not self.__has_regridder(regridder_class, method):
            self.__create_regridder(regridder_class, method)

        return self.__get_existing_regridder(regridder_class, method)


def assign_coord_if_present(
    coordname: str, target_grid: GridDataArray, maybe_has_coords_attr: Any
):
    """
    If ``maybe_has_coords`` has a ``coords`` attribute and if coordname in
    target_grid, copy coord.
    """
    if coordname in target_grid.coords:
        if coordname in target_grid.coords and hasattr(maybe_has_coords_attr, "coords"):
            maybe_has_coords_attr = maybe_has_coords_attr.assign_coords(
                {coordname: target_grid.coords[coordname].values[()]}
            )
    return maybe_has_coords_attr


def _regrid_array(
    package: IRegridPackage,
    varname: str,
    regridder_collection: RegridderInstancesCollection,
    regridder_name: str,
    regridder_function: str,
    target_grid: GridDataArray,
) -> Optional[GridDataArray]:
    """
    Regrids a data_array. The array is specified by its key in the dataset.
    Each data-array can represent:
    -a scalar value, valid for the whole grid
    -an array of a different scalar per layer
    -an array with a value per grid block
    -None
    """

    # skip regridding for arrays with no valid values (such as "None")
    if not package._valid(package.dataset[varname].values[()]):
        return None

    # the dataarray might be a scalar. If it is, then it does not need regridding.
    if is_scalar(package.dataset[varname]):
        return package.dataset[varname].values[()]

    if isinstance(package.dataset[varname], xr.DataArray):
        coords = package.dataset[varname].coords
        # if it is an xr.DataArray it may be layer-based; then no regridding is needed
        if not ("x" in coords and "y" in coords):
            return package.dataset[varname]

        # if it is an xr.DataArray it needs the dx, dy coordinates for regridding, which are otherwise not mandatory
        if not ("dx" in coords and "dy" in coords):
            raise ValueError(
                f"DataArray {varname} does not have both a dx and dy coordinates"
            )

    # obtain an instance of a regridder for the chosen method
    regridder = regridder_collection.get_regridder(
        regridder_name,
        regridder_function,
    )

    # store original dtype of data
    original_dtype = package.dataset[varname].dtype

    # regrid data array
    regridded_array = regridder.regrid(package.dataset[varname])

    # reconvert the result to the same dtype as the original
    return regridded_array.astype(original_dtype)


def _get_unique_regridder_types(model: IModel) -> defaultdict[RegridderType, list[str]]:
    """
    This function loops over the packages and  collects all regridder-types that are in use.
    """
    methods: defaultdict = defaultdict(list)
    for _, pkg in model.items():
        if isinstance(pkg, IRegridPackage):
            pkg_methods = pkg.get_regrid_methods()
            for variable in pkg_methods:
                if (
                    variable in pkg.dataset.data_vars
                    and pkg.dataset[variable].values[()] is not None
                ):
                    regriddertype = pkg_methods[variable][0]
                    functiontype = pkg_methods[variable][1]
                    if functiontype not in methods[regriddertype]:
                        methods[regriddertype].append(functiontype)
    return methods


@typedispatch  # type: ignore[no-redef]
def _regrid_like(
    package: IRegridPackage,
    target_grid: GridDataArray,
    regridder_types: Optional[dict[str, tuple[RegridderType, str]]] = None,
) -> IPackage:
    """
    Creates a package of the same type as this package, based on another discretization.
    It regrids all the arrays in this package to the desired discretization, and leaves the options
    unmodified. At the moment only regridding to a different planar grid is supported, meaning
    ``target_grid`` has different ``"x"`` and ``"y"`` or different ``cell2d`` coords.

    The regridding methods can be specified in the _regrid_method attribute of the package. These are the defaults
    that specify how each array should be regridded. These defaults can be overridden using the input
    parameters of this function.

    Examples
    --------
    To regrid the npf package with a non-default method for the k-field, call regrid_like with these arguments:

    >>> new_npf = npf.regrid_like(like, {"k": (imod.RegridderType.OVERLAP, "mean")})


    Parameters
    ----------
    target_grid: xr.DataArray or xu.UgridDataArray
        a grid defined over the same discretization as the one we want to regrid the package to
    regridder_types: dict(str->(regridder type,str))
        dictionary mapping arraynames (str) to a tuple of regrid type (a specialization class of BaseRegridder) and function name (str)
        this dictionary can be used to override the default mapping method.

    Returns
    -------
    a package with the same options as this package, and with all the data-arrays regridded to another discretization,
    similar to the one used in input argument "target_grid"
    """
    if not hasattr(package, "_regrid_method"):
        raise NotImplementedError(
            f"Package {type(package).__name__} does not support regridding"
        )

    if hasattr(package, "auxiliary_data_fields"):
        remove_expanded_auxiliary_variables_from_dataset(package)

    regridder_collection = RegridderInstancesCollection(
        package.dataset, target_grid=target_grid
    )

    regridder_settings = package.get_regrid_methods()
    if regridder_types is not None:
        regridder_settings.update(regridder_types)

    new_package_data = package.get_non_grid_data(list(regridder_settings.keys()))

    for (
        varname,
        regridder_type_and_function,
    ) in regridder_settings.items():
        regridder_name, regridder_function = regridder_type_and_function

        # skip variables that are not in this dataset
        if varname not in package.dataset.keys():
            continue

        # regrid the variable
        new_package_data[varname] = _regrid_array(
            package,
            varname,
            regridder_collection,
            regridder_name,
            regridder_function,
            target_grid,
        )
        # set dx and dy if present in target_grid
        new_package_data[varname] = assign_coord_if_present(
            "dx", target_grid, new_package_data[varname]
        )
        new_package_data[varname] = assign_coord_if_present(
            "dy", target_grid, new_package_data[varname]
        )
    if hasattr(package, "auxiliary_data_fields"):
        expand_transient_auxiliary_variables(package)

    return package.__class__(**new_package_data)


@typedispatch  # type: ignore[no-redef]
def _regrid_like(
    model: IModel, target_grid: GridDataArray, validate: bool = True
) -> IModel:
    """
    Creates a model by regridding the packages of this model to another discretization.
    It regrids all the arrays in the package using the default regridding methods.
    At the moment only regridding to a different planar grid is supported, meaning
    ``target_grid`` has different ``"x"`` and ``"y"`` or different ``cell2d`` coords.

    Parameters
    ----------
    target_grid: xr.DataArray or xu.UgridDataArray
        a grid defined over the same discretization as the one we want to regrid the package to
    validate: bool
        set to true to validate the regridded packages

    Returns
    -------
    a model with similar packages to the input model, and with all the data-arrays regridded to another discretization,
    similar to the one used in input argument "target_grid"
    """
    supported, error_with_object_name = model.is_regridding_supported()
    if not supported:
        raise ValueError(
            f"regridding this model cannot be done due to the presence of package {error_with_object_name}"
        )
    new_model = model.__class__()

    for pkg_name, pkg in model.items():
        if isinstance(pkg, (IRegridPackage, ILineDataPackage, IPointDataPackage)):
            new_model[pkg_name] = pkg.regrid_like(target_grid)
        else:
            raise NotImplementedError(
                f"regridding is not implemented for package {pkg_name} of type {type(pkg)}"
            )

    methods = _get_unique_regridder_types(model)
    output_domain = _get_regridding_domain(model, target_grid, methods)
    new_model.mask_all_packages(output_domain)
    new_model.purge_empty_packages()
    if validate:
        status_info = NestedStatusInfo("Model validation status")
        status_info.add(new_model.validate("Regridded model"))
        if status_info.has_errors():
            raise ValidationError("\n" + status_info.to_string())
    return new_model


@typedispatch  # type: ignore[no-redef]
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
            result[key] = item.regrid_like(target_grid, validate)
        elif key == "gwtgwf_exchanges":
            pass
        elif isinstance(item, IPackage) and not isinstance(item, IRegridPackage):
            result[key] = copy.deepcopy(item)

        else:
            raise NotImplementedError(f"regridding not supported for {key}")

    return result


@typedispatch  # type: ignore[no-redef]
def _regrid_like(
    package: ILineDataPackage, target_grid: GridDataArray, *_
) -> ILineDataPackage:
    """
    The regrid_like method is irrelevant for this package as it is
    grid-agnostic, instead this method clips the package based on the grid
    exterior.
    """
    return clip_by_grid(package, target_grid)


@typedispatch  # type: ignore[no-redef]
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


@typedispatch  # type: ignore[no-redef]
def _regrid_like(package: object, target_grid: GridDataArray, *_) -> None:
    raise TypeError("this object cannot be regridded")


def _get_regridding_domain(
    model: IModel,
    target_grid: GridDataArray,
    methods: defaultdict[RegridderType, list[str]],
) -> GridDataArray:
    """
    This method computes the output-domain for a regridding operation by regridding idomain with
    all regridders. Each regridder may leave some cells inactive. The output domain for the model consists of those
    cells that all regridders consider active.
    """
    idomain = model.domain
    regridder_collection = RegridderInstancesCollection(
        idomain, target_grid=target_grid
    )
    included_in_all = ones_like(target_grid)
    regridders = [
        regridder_collection.get_regridder(regriddertype, function)
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
