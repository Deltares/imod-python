from typing import Dict
from xugrid.regrid.regridder import (
    BaseRegridder,
    BarycentricInterpolator,
    CentroidLocatorRegridder,
    OverlapRegridder,
    RelativeOverlapRegridder,
)


def create_regridder_from_string(name, source_grid, target_grid, method=None)->BaseRegridder:
    """
    This function creates a regridder.

    Parameters
    ----------
    name:  name of the regridder (for example, "CentroidLocatorRegridder")
    source_grid: (ugrid or xarray) data-array containing the discretization of the source grid as coordinates
    target_grid: (ugrid or xarray) data-array containing the discretization of the targetr-grid as coordinates
    method: optionally, method used for regridding ( for example, "geometric_mean").

    Returns
    -------
    a regridder of the desired type and that uses the desired function

    """
    regridder = None

    # verify method is None for regridders that don't support methods
    if name == "BarycentricInterpolator" or name == "CentroidLocatorRegridder":
        if method is not None:
            raise ValueError(f"{name} does not support methods")

    if name == "BarycentricInterpolator":
        regridder = BarycentricInterpolator(source_grid, target_grid)
    elif name == "OverlapRegridder":
        regridder = OverlapRegridder(source_grid, target_grid, method)
    elif name == "RelativeOverlapRegridder":
        regridder = RelativeOverlapRegridder(source_grid, target_grid, method)
    elif name == "CentroidLocatorRegridder":
        regridder = CentroidLocatorRegridder(source_grid, target_grid)

    if regridder is not None:
        return regridder

    raise ValueError("unknown regridder type " + name)


class RegridderInstancesCollection:
    """
    This class stores any number of regridders that can regrid a single source grid to a single target grid.
    By storing the regridders, we make sure the regridders can be re-used for different arrays on the same grid.
    This is important because computing the regridding weights is a costly affair.
    """

    def __init__(self, source_grid, target_grid) -> None:
        self.regridder_instances = {}
        self._source_grid = source_grid
        self._target_grid = target_grid

    def __has_regridder(self, name, method)->bool:
        return (name, method) in self.regridder_instances.keys()

    def __get_existing_regridder(self, name, method)->BaseRegridder:
        if self.__has_regridder(name, method):
            return self.regridder_instances[(name, method)]
        raise ValueError("no existing regridder of type " + name)

    def __create_regridder(self, name, method)->BaseRegridder:
        self.regridder_instances[(name, method)] = create_regridder_from_string(
            name, self._source_grid, self._target_grid, method
        )
        return self.regridder_instances[(name, method)]

    def get_regridder(self, name, method=None)->BaseRegridder:
        """
        returns a regridder of the specified type-name and with the specified method.
        """
        if not self.__has_regridder(name, method):
            self.__create_regridder(name, method)

        return self.__get_existing_regridder(name, method)


def get_non_grid_data(package, grid_names)->Dict[str, any]:
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
        all_non_grid_data.remove(name)
    for name in all_non_grid_data:
        result[name] = package.dataset[name].values[()]
    return result
