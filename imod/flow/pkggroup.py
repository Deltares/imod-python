import abc
import collections
import enum

from imod.flow.pkgbase import Vividict
from imod.flow.timeutil import insert_unique_package_times


class PackageGroup(collections.UserDict, abc.ABC):
    """
    Groups for packs that support multiple systems:
    * chd
    * drn
    * ghb
    * riv
    * wel
    """

    __slots__ = ["_n_variables"]

    def __init__(self, **kwargs):
        collections.UserDict.__init__(self)
        for k, v in kwargs.items():
            self[k] = v
        # self.reorder_keys()

    def compose(
        self, directory, globaltimes, nlayer, compose_projectfile=True, composition=None
    ):

        pkggroup_times = []
        pkggroup_times, _ = insert_unique_package_times(self.items(), pkggroup_times)

        if composition is None:
            composition = Vividict()

        for i, (key, pkg) in enumerate(self.items()):
            system_index = i + 1
            pkg.compose(
                directory.joinpath(key),
                globaltimes,
                nlayer,
                system_index=system_index,
                compose_projectfile=compose_projectfile,
                composition=composition,
                pkggroup_time=pkggroup_times,
            )

        return composition


class ConstantHeadGroup(PackageGroup):
    _n_variables = 1


class DrainageGroup(PackageGroup):
    _n_variables = 2


class GeneralHeadBoundaryGroup(PackageGroup):
    _n_variables = 2


class RiverGroup(PackageGroup):
    _n_variables = 4


class WellGroup(PackageGroup):
    _n_variables = 1


# dict might be easier than Enumerator...
class PackageGroups(enum.Enum):
    chd = ConstantHeadGroup
    drn = DrainageGroup
    ghb = GeneralHeadBoundaryGroup
    riv = RiverGroup
    wel = WellGroup
