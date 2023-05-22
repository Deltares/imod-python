from xugrid.regrid.regridder import (
    BarycentricInterpolator,
    CentroidLocatorRegridder,
    OverlapRegridder,
    RelativeOverlapRegridder,
)


def create_regridder_from_string(name, source, target, method=None):
    """
    This function creates a regridder.

    Parameters
    ----------
    name:  name of the regridder (for example, "CentroidLocatorRegridder")
    source: (ugrid or xarray) data-array containing the discretization of the source grid as coordinates
    target: (ugrid or xarray) data-array containing the discretization of the targetr-grid as coordinates
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
        regridder = BarycentricInterpolator(source, target)
    elif name == "OverlapRegridder":
        regridder = OverlapRegridder(source, target, method)
    elif name == "RelativeOverlapRegridder":
        regridder = RelativeOverlapRegridder(source, target, method)
    elif name == "CentroidLocatorRegridder":
        regridder = CentroidLocatorRegridder(source, target)

    if regridder is not None:
        return regridder

    raise ValueError("unknown regridder type " + name)


class RegridderInstancesCollection:
    """
    This class stores any number of regridders that can regrid a single source grid to a single target grid.
    By storing the regridders, we make sure the regridders can be re-used for different arrays on the same grid.
    This is important because computing the regridding weights is a costly affair.
    """

    def __init__(self, source, target) -> None:
        self.regridder_instances = {}
        self._source = source
        self._target = target

    def _has_regridder(self, name, method):
        return (name, method) in self.regridder_instances.keys()

    def _get_existing_regridder(self, name, method):
        if self._has_regridder(name, method):
            return self.regridder_instances[(name, method)]
        raise ValueError("no existing regridder of type " + name)

    def _create_regridder(self, name, method):
        self.regridder_instances[(name, method)] = create_regridder_from_string(
            name, self._source, self._target, method
        )
        return self.regridder_instances[(name, method)]

    def get_regridder(self, name, method=None):
        """
        returns a regridder of the specified type-name and with the specified method.
        """
        if not self._has_regridder(name, method):
            self._create_regridder(name, method)

        return self._get_existing_regridder(name, method)


def get_non_grid_data(package, grid_names):
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
