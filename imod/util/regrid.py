import abc
from enum import Enum
from typing import Dict, Optional, Tuple, TypeAlias, Union

import xugrid as xu
from xugrid.regrid.regridder import BaseRegridder

from imod.typing import GridDataArray
from imod.typing.grid import get_grid_geometry_hash


class RegridderType(Enum):
    """
    Enumerator referring to regridder types in ``xugrid``. These can be used
    safely in scripts, remaining backwards compatible for when it is decided to
    rename regridders in ``xugrid``. For an explanation what each regridder type
    does, we refer to the `xugrid documentation
    <https://deltares.github.io/xugrid/examples/regridder_overview.html>`_

    Examples
    --------
    You can use this as follows:

    >>> regridder = RegridderType.OVERLAP(source, target, method="mean")
    >>> result = regridder.regrid(uda)

    This is equivalent to:

    >>> regridder = xu.OverlapRegridder(source, target, method="mean")
    >>> result = regridder.regrid(uda)
    """

    CENTROIDLOCATOR = xu.CentroidLocatorRegridder
    BARYCENTRIC = xu.BarycentricInterpolator
    OVERLAP = xu.OverlapRegridder
    RELATIVEOVERLAP = xu.RelativeOverlapRegridder


HashRegridderMapping = Tuple[int, int, BaseRegridder]
RegridVarType: TypeAlias = Tuple[RegridderType, str] | Tuple[RegridderType]


class RegridderWeightsCache:
    """
    This class stores any number of regridders that can regrid a single source
    grid to a single target grid. By storing the regridders, we make sure the
    regridders can be re-used for different arrays on the same grid. Regridders
    are stored based on their type (`see these
    docs <https://deltares.github.io/xugrid/examples/regridder_overview.html>`_)
    and planar coordinates (x, y). This is important because computing the
    regridding weights is a costly affair.

    Parameters
    ----------
    max_cache_size: int
        The maximum number of regridders that can be stored in the cache. If
        the cache is full, the oldest regridder will be removed.

    Examples
    --------
    Different method, regridder weights are reused.

    >>> cache = imod.util.RegridderWeightsCache()
    >>> regridder1 = cache.get_regridder(source, target, RegridderType.OVERLAP, "mean")
    >>> regridder2 = cache.get_regridder(source, target, RegridderType.OVERLAP, "mode")
    >>> print(cache.weights_cache)

    Different regridder type, different regridder weights.

    >>> cache = imod.util.RegridderWeightsCache()
    >>> regridder1 = cache.get_regridder(source, target, RegridderType.OVERLAP, "mean")
    >>> regridder2 = cache.get_regridder(source, target, RegridderType.RELATIVEOVERLAP, "mean")
    >>> print(cache.weights_cache)

    Different source geometries, different regridder weights.

    >>> cache = imod.util.RegridderWeightsCache()
    >>> regridder1 = cache.get_regridder(source_xy1, target, RegridderType.OVERLAP, "mean")
    >>> regridder2 = cache.get_regridder(source_xy2, target, RegridderType.OVERLAP, "mean")
    >>> print(cache.weights_cache)

    Different source grids with same geometry, regridder weights are reused.

    >>> cache = imod.util.RegridderWeightsCache()
    >>> source2 = xr.ones_like(source)
    >>> regridder1 = cache.get_regridder(source, target, RegridderType.OVERLAP, "mean")
    >>> regridder2 = cache.get_regridder(source2, target, RegridderType.OVERLAP, "mean")
    >>> print(cache.weights_cache)
    """

    def __init__(
        self,
        max_cache_size: int = 6,
    ) -> None:
        self.regridder_instances: dict[
            tuple[type[BaseRegridder], Optional[str]], BaseRegridder
        ] = {}
        self.weights_cache: Dict[HashRegridderMapping, GridDataArray] = {}
        self.max_cache_size = max_cache_size

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
        source_grid: GridDataArray,
        target_grid: GridDataArray,
        regridder_type: Union[RegridderType, BaseRegridder],
        method: Optional[str] = None,
    ) -> BaseRegridder:
        """
        Returns a regridder of the specified type and with the specified method.
        The desired type can be passed through the argument "regridder_type" as
        an enumerator or as a class. The following two are equivalent:

        >>> cache.get_regridder(RegridderType.OVERLAP, "mean")
        >>> cache.get_regridder(xu.OverlapRegridder, "mean")

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

        if "layer" not in source_grid.coords and "layer" in target_grid.coords:
            target_grid = target_grid.drop_vars("layer")

        source_hash = get_grid_geometry_hash(source_grid)
        target_hash = get_grid_geometry_hash(target_grid)
        key = (source_hash, target_hash, regridder_class)
        if key not in self.weights_cache.keys():
            if len(self.weights_cache) >= self.max_cache_size:
                self._remove_first_regridder()
            kwargs = {"source": source_grid, "target": target_grid}
            if method is not None:
                kwargs["method"] = method
            regridder = regridder_class(**kwargs)
            self.weights_cache[key] = regridder.weights
        else:
            kwargs = {"weights": self.weights_cache[key], "target": target_grid}
            if method is not None:
                kwargs["method"] = method
            regridder = regridder_class.from_weights(**kwargs)

        return regridder

    def _remove_first_regridder(self):
        keys = list(self.weights_cache.keys())
        self.weights_cache.pop(keys[0])
