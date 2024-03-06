import abc

from typing import Any, Optional, Union
import xarray as xr
import xugrid as xu
from xugrid.regrid.regridder import BaseRegridder
from fastcore.dispatch import typedispatch
from imod.typing.grid import GridDataArray
from imod.mf6.utilities.clip import clip_by_grid
from imod.mf6.interfaces.ilinedatapackage import ILineDataPackage
from imod.mf6.interfaces.ipointdatapackage import IPointDataPackage
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.interfaces.ipackage import IPackage
from imod.mf6.utilities.regridding_types import RegridderType
import copy
from xarray.core.utils import is_scalar

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
            Tuple[type[BaseRegridder], Optional[str]], BaseRegridder
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


def get_non_grid_data(package, grid_names: list[str]) -> dict[str, Any]:
    """
    This function copies the attributes of a dataset that are scalars, such as options.

    parameters
    ----------
    grid_names: list of str
        the names of the attribbutes of a dataset that are grids.
    """
    result = {}
    all_non_grid_data = list(package.dataset.keys())
    for name in grid_names:
        if name in all_non_grid_data:
            all_non_grid_data.remove(name)
    for name in all_non_grid_data:
        if "time" in package.dataset[name].coords:
            result[name] = package.dataset[name]
        else:
            result[name] = package.dataset[name].values[()]
    return result


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

@typedispatch  # type: ignore[no-redef]
def regrid_like(
    package: ILineDataPackage, target_grid: GridDataArray, *_) -> ILineDataPackage:
    """
    The regrid_like method is irrelevant for this package as it is
    grid-agnostic, instead this method clips the package based on the grid
    exterior.
    """
    return clip_by_grid(package, target_grid)

@typedispatch  # type: ignore[no-redef]
def regrid_like(
    package: IPointDataPackage, target_grid: GridDataArray, *_
) -> IPointDataPackage:
    """
    The regrid_like method is irrelevant for this package as it is
    grid-agnostic, instead this method clips the package based on the grid
    exterior.
    """
    target_grid_2d = target_grid.isel(layer=0, drop=True, missing_dims="ignore")
    return clip_by_grid(package, target_grid_2d)

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

    regridder_collection = RegridderInstancesCollection(
        package.dataset, target_grid=target_grid
    )

    regridder_settings = copy.deepcopy(package._regrid_method)
    if regridder_types is not None:
        regridder_settings.update(regridder_types)

    new_package_data = get_non_grid_data(package, list(regridder_settings.keys()))

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

    return package.__class__(**new_package_data)
