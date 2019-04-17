from collections import UserDict
from enum import Enum


class PackageGroup(UserDict):
    """
    Groups for packes that support multiple systems:
    * chd
    * drn
    * ghb
    * riv
    * wel
    """

    def __init__(self, **kwargs):
        UserDict.__init__(self)
        for k, v in kwargs.items():
            self[k] = v
        self.key_order()

    def key_order(self):
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
            raise ValueError("Only one system with concentrations allowed per package")
        self.first_key = order[0]
        self.key_order = order

    def max_n_sinkssources(self):
        # TODO: check if this is necessary
        # with sinks, are system 2 and higher also a sink?
        # If that's the case, active_max_n is sufficient for every package
        # key = self.first_key
        # ds = self[key]

        # if "time" in ds[varname].coords:
        #    nmax = int(ds[varname].groupby("time").count().max())
        # else:
        #    nmax = int(ds[varname].count())
        return self.n_max_active

    def render(self, directory, globaltimes):
        d = {}
        d["n_systems"] = len(self.keys())
        self.n_max_active = sum(
            [v._max_active_n(self._cellcount_varname) for v in self.values()]
        )
        d["n_max_active"] = self.n_max_active
        d["save_budget"] = any([v["save_budget"] for v in self.values()])

        content = [self._template.format(**d)]
        for i, key in enumerate(self.key_order):
            system_index = i + 1
            content.append(
                self[key]._render(
                    directory=directory,
                    globaltimes=globaltimes,
                    system_index=system_index,
                )
            )
        return "".join(content)

    def render_ssm(self, directory, globaltimes):
        # Only render for the first system, that has concentrations defined.
        key = self.first_key
        return self[key]._ssm_render(directory, globaltimes)


class ConstantHeadGroup(PackageGroup):
    _cellcount_varname = "head"
    _template = "[chd]\n" "    mchdsys = {n_systems}\n" "    mxactc = {n_max_active}\n"


class DrainageGroup(PackageGroup):
    _cellcount_varname = "conductance"
    _template = (
        "[ghb]\n"
        "    mdrnsys = {n_systems}\n"
        "    mxactd = {n_max_active}\n"
        "    idrncb = {save_budget}"
    )


class GeneralHeadBoundaryGroup(PackageGroup):
    _cellcount_varname = "conductance"
    _template = (
        "[ghb]\n"
        "    mghbsys = {n_systems}\n"
        "    mxactb = {n_max_active}\n"
        "    ighbcb = {save_budget}"
    )


class RiverGroup(PackageGroup):
    _cellcount_varname = "conductance"
    _template = (
        "[riv]\n"
        "    mrivsys = {n_systems}\n"
        "    mxactr = {n_max_active}\n"
        "    irivcb = {save_budget}"
    )


# dict might be easier than Enumerator...
class PackageGroups(Enum):
    chd = ConstantHeadGroup
    drn = DrainageGroup
    ghb = GeneralHeadBoundaryGroup
    riv = RiverGroup
