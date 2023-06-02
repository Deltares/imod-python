from enum import Enum
from typing import Dict, List, Union

import xarray as xr
import xugrid as xu
from xugrid.regrid.regridder import (
    BaseRegridder,
)
class RegridderType(Enum):
    CENTROIDLOCATOR = xu.CentroidLocatorRegridder
    BARYCENTRIC = xu.BarycentricInterpolator
    OVERLAP = xu.OverlapRegridder
    RELATIVEOVERLAP = xu.RelativeOverlapRegridder


    def create_regridder(self,  source_grid: Union[xr.Dataset, xr.DataArray, xu.UgridDataset, xu.UgridDataArray],
        target_grid: Union[xr.DataArray, xu.UgridDataArray],
        method: str = None,
    ) -> BaseRegridder:
        """
        This function creates a regridder.

        Parameters
        ----------
        regridder_type: RegridderType
            enumerator for thet type of regridder
        source_grid: xr.Dataset, xr.DataArray , xu.UgridDataset,   xu.UgridDataArray
            data-array or dataset containing the discretization of the source grid as coordinates
        target_grid:  xr.DataArray of  xu.UgridDataArray
            data-array containing the discretization of the targetr-grid as coordinates
        method: str
            optionally, method used for regridding ( for example, "geometric_mean").

        Returns
        -------
        a regridder of the desired type and that uses the desired function

        """
        regridder = None
        try:    
            if method is None:
                regridder = self.value(source_grid, target_grid)
            else:
                regridder = self.value(source_grid, target_grid, method)
        except:
            raise ValueError(f"failed to create a regridder {self.value} with method {method}")
        return regridder
      

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

    def __has_regridder(self, regridder_type: RegridderType, method: str) -> bool:
        return (regridder_type, method) in self.regridder_instances.keys()

    def __get_existing_regridder(self, regridder_type: RegridderType, method: str) -> BaseRegridder:
        if self.__has_regridder(regridder_type, method):
            return self.regridder_instances[(regridder_type, method)]
        raise ValueError("no existing regridder of type " + str(regridder_type))

    def __create_regridder(self, regridder_type: RegridderType, method: str) -> BaseRegridder:
        self.regridder_instances[(regridder_type, method)] = regridder_type.create_regridder(
            self._source_grid, self._target_grid, method
        )
        return self.regridder_instances[(regridder_type, method)]

    def get_regridder(self, regridder_type: RegridderType, method: str = None) -> BaseRegridder:
        """
        returns a regridder of the specified type-name and with the specified method.
        """
        if not self.__has_regridder(regridder_type, method):
            self.__create_regridder(regridder_type, method)

        return self.__get_existing_regridder(regridder_type, method)


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
