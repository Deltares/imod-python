import abc
import collections
import enum


class PackageGroup(collections.UserDict, abc.ABC):
    """
    Groups for packes that support multiple systems:
    * chd
    * drn
    * ghb
    * riv
    * wel
    """

    def __init__(self, **kwargs):
        collections.UserDict.__init__(self)
        for k, v in kwargs.items():
            self[k] = v
        self.reorder_keys()

    def reorder_keys(self):
        """
        Order packages so that the one with with concentration is first.
        Check whether concentration for only one system has been defined.
        """
        n_system_concentrations = 0
        order = []
        for k, v in self.items():
            if "concentration" in v.data_vars:
                n_system_concentrations += 1
                order.insert(0, k)
            else:
                order.append(k)
        if n_system_concentrations > 1:
            raise ValueError(
                f"Multiple systems with concentrations detected: {order}\n"
                "Only one system with concentration is allowed per package kind."
            )
        self.first_key = order[0]
        self.key_order = order

    def max_n_sinkssources(self):
        return sum(pkg._ssm_cellcount for pkg in self.values())

    def render(self, directory, globaltimes, nlayer, nrow, ncol):
        d = {}
        d["n_systems"] = len(self.keys())
        d["n_max_active"] = sum(
            [
                v._max_active_n(self._cellcount_varname, nlayer, nrow, ncol)  # pylint:disable=no-member
                for v in self.values()
            ]
        )
        d["save_budget"] = 1 if any([v.save_budget for v in self.values()]) else 0

        content = [self._template.format(**d)]  # pylint: disable=no-member
        for i, key in enumerate(self.key_order):
            system_index = i + 1
            content.append(
                self[key]._render(
                    directory=directory.joinpath(key),
                    globaltimes=globaltimes,
                    system_index=system_index,
                )
            )
        return "".join(content)

    def render_ssm(self, directory, globaltimes):
        # Only render for the first system, that has concentrations defined.
        key = self.first_key
        return self[key]._render_ssm(directory.joinpath(key), globaltimes)


class ConstantHeadGroup(PackageGroup):
    _cellcount_varname = "head"
    _template = "[chd]\n" "    mchdsys = {n_systems}\n" "    mxactc = {n_max_active}\n"


class DrainageGroup(PackageGroup):
    _cellcount_varname = "elevation"
    _template = (
        "[drn]\n"
        "    mdrnsys = {n_systems}\n"
        "    mxactd = {n_max_active}\n"
        "    idrncb = {save_budget}"
    )


class GeneralHeadBoundaryGroup(PackageGroup):
    _cellcount_varname = "head"
    _template = (
        "[ghb]\n"
        "    mghbsys = {n_systems}\n"
        "    mxactb = {n_max_active}\n"
        "    ighbcb = {save_budget}"
    )


class RiverGroup(PackageGroup):
    _cellcount_varname = "stage"
    _template = (
        "[riv]\n"
        "    mrivsys = {n_systems}\n"
        "    mxactr = {n_max_active}\n"
        "    irivcb = {save_budget}"
    )


class WellGroup(PackageGroup):
    _cellcount_varname = None
    _template = (
        "[wel]\n"
        "    mwelsys = {n_systems}\n"
        "    mxactw = {n_max_active}\n"
        "    iwelcb = {save_budget}"
    )


# dict might be easier than Enumerator...
class PackageGroups(enum.Enum):
    chd = ConstantHeadGroup
    drn = DrainageGroup
    ghb = GeneralHeadBoundaryGroup
    riv = RiverGroup
    wel = WellGroup
