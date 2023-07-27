import abc
from enum import Enum
from typing import Dict, List, Optional, Union

import xarray as xr
import xugrid as xu
from xugrid.regrid.regridder import BaseRegridder

from imod.typing.grid import GridDataArray


class RegridderType(Enum):
    """
    Enumerator referring to regridder types in ``xugrid``.
    These can be used safely in scripts, remaining backwards compatible for
    when it is decided to rename regridders in ``xugrid``. For an explanation
    what each regridder type does, we refer to the `xugrid documentation <https://deltares.github.io/xugrid/examples/regridder_overview.html>`_
    """

    CENTROIDLOCATOR = xu.CentroidLocatorRegridder
    BARYCENTRIC = xu.BarycentricInterpolator
    OVERLAP = xu.OverlapRegridder
    RELATIVEOVERLAP = xu.RelativeOverlapRegridder


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
        self.regridder_instances = {}
        self._source_grid = source_grid
        self._target_grid = target_grid

    def __has_regridder(
        self, regridder_type: abc.ABCMeta, method: Optional[str] = None
    ) -> bool:
        return (regridder_type, method) in self.regridder_instances.keys()

    def __get_existing_regridder(
        self, regridder_type: abc.ABCMeta, method: Optional[str]
    ) -> BaseRegridder:
        if self.__has_regridder(regridder_type, method):
            return self.regridder_instances[(regridder_type, method)]
        raise ValueError("no existing regridder of type " + str(regridder_type))

    def __create_regridder(
        self, regridder_type: abc.ABCMeta, method: Optional[str]
    ) -> BaseRegridder:
        if method is None:
            method_args = ()
        else:
            method_args = (method,)

        self.regridder_instances[(regridder_type, method)] = regridder_type(
            self._source_grid, self._target_grid, *method_args
        )
        return self.regridder_instances[(regridder_type, method)]

    def __get_regridder_class(self, regridder_type: RegridderType) -> abc.ABCMeta:
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
        regridder_type: Union[RegridderType, abc.ABCMeta],
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


def get_non_grid_data(package, grid_names: List[str]) -> Dict[str, any]:
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
        result[name] = package.dataset[name].values[()]
    return result


def align_grid_coordinates(
    regridded_array: GridDataArray, target_grid: GridDataArray
) -> GridDataArray:
    """'
    The regridded array may have coordinates that are not exactly the same as those of the target grid
    due to rounding errors (coordinates are re-computed in xugrid based on dx, dy).
    We overwrite the regridded coordinates with the target grid coordinates.
    This is not necessary for integer coordinates, such as layer and those of unstructured grids.
    """
    expected_coordinates = ["dx", "dy", "x", "y"]
    for coord in expected_coordinates:
        if coord in regridded_array.coords:
            regridded_array = regridded_array.assign_coords(
                {coord: target_grid.coords[coord].values[()]}
            )
    return regridded_array
        )
    if "x" in regridded_array.coords:
        regridded_array = regridded_array.assign_coords(
            {"x": target_grid.coords["x"].values[()]}
        )
    if "y" in regridded_array.coords:
        regridded_array = regridded_array.assign_coords(
            {"y": target_grid.coords["y"].values[()]}
        )
    return regridded_array
