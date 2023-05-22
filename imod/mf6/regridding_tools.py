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
    name:  name of the regridder (for example, CentroidLocatorRegridder)
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
    def __init__(self) -> None:
        self.regridder_instances = {}

    def _has_regridder(self, name, method):
        return (name, method) in self.regridder_instances.keys()

    def _get_existing_regridder(self, name, method):
        if self._has_regridder(name, method):
            return self.regridder_instances[(name, method)]
        raise ValueError("no existing regridder of type " + name)

    def _create_regridder(self, name, method, source, target):
        self.regridder_instances[(name, method)] = create_regridder_from_string(
            name, source, target, method
        )
        return self.regridder_instances[(name, method)]

    def get_regridder(self, name, source, target, method=None):
        if not self._has_regridder(name, method):
            self._create_regridder(name, method, source, target)

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
