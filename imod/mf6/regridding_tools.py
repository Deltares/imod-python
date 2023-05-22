from xugrid.regrid.regridder import (
    BarycentricInterpolator,
    OverlapRegridder,
    RelativeOverlapRegridder,
)


def create_regridder_from_string(name: str, source, target):
    regridder = None
    if name == "BarycentricInterpolator":
        regridder = BarycentricInterpolator(source, target)
    elif name == "OverlapRegridder":
        regridder = OverlapRegridder(source, target)
    elif name == "RelativeOverlapRegridder":
        regridder = RelativeOverlapRegridder(source, target)

    if regridder is not None:
        return regridder
    raise ValueError("unkwown regridder type " + name)


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
            name,method, source, target
        )
        return self.regridder_instances[(name,method)]

    def get_regridder(self, name, method, source, target):
        if not self._has_regridder(name):
            self._create_regridder(name, source, target)

        return self._get_existing_regridder(name)


def get_non_grid_data(package, grid_names):
    result = {}
    all_non_grid_data = list(package.dataset.keys())
    for name in grid_names:
        all_non_grid_data.remove(name)
    for name in all_non_grid_data:
        result[name] = package.dataset[name].values[()]
    return result
