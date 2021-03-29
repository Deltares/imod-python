import abc
import collections
import enum
import jinja2

from imod.flow.util import Vividict


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

    _template = jinja2.Template("{{n_systems * n_layer}}, ({{_pkg_id}})")

    _template_projectfile = jinja2.Template("{{n_variables}}, {{n_systems * n_layer}}")

    def __init__(self, **kwargs):
        collections.UserDict.__init__(self)
        for k, v in kwargs.items():
            self[k] = v
        # self.reorder_keys()

    def compose(
        self, directory, globaltimes, compose_projectfile=True, composition=None
    ):

        if composition is None:
            composition = Vividict()

        for i, (key, pkg) in enumerate(self.items()):
            sys_nr = i + 1
            pkg.compose(
                directory.joinpath(key),
                globaltimes,
                sys_nr=sys_nr,
                compose_projectfile=compose_projectfile,
                composition=composition,
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
