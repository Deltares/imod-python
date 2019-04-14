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
        for k, v in kwargs.items():
            self[k] = v
        self.first_key = self.first_package()

    def first_package(self):
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
        return order[0]

    def max_n_sinkssources(self):
        # TODO: check if this is necessary
        # with sinks, are system 2 and higher also a sink?
        # If that's the case, active_max_n is sufficient for every package
        key = self.first_key
        ds = self[key]
        if "time" in ds[varname].coords:
            nmax = int(ds[varname].groupby("time").count().max())
        else:
            nmax = int(ds[varname].count())
        return nmax

    def render(self, directory, globaltimes):
        d = {}
        d["n_systems"] = len(self.keys())
        d["nmax_active"] = sum([v._max_active_n() for v in self.values()])
        d["save_budget"] = any([v["save_budget"] for v in self.values()])

        content = [self._template.render(d)]
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

    def ssm_render(self, directory, globaltimes):
        # Only render for the first system, that has concentrations defined.
        key = self.first_key
        return self[key]._ssm_render(directory, globaltimes)


class ConstantHeadGroup(PackageGroup):
    _template = "[chd]\n" "    mchdsys = {n_systems}\n" "    mxactc = {n_max_active}\n"


class DrainageGroup(PackageGroup):
    _template = (
        "[ghb]\n"
        "    mdrnsys = {n_systems}\n"
        "    mxactd = {n_max_active}\n"
        "    idrncb = {save_budget}\n"
    )


class GeneralHeadBoundaryGroup(PackageGroup):
    _template = (
        "[ghb]\n"
        "    mghbsys = {n_systems}\n"
        "    mxactb = {n_max_active}\n"
        "    ighbcb = {save_budget}\n"
    )


class RiverGroup(PackageGroup):
    _template = (
        "[riv]\n"
        "    mrivsys = {n_systems}\n"
        "    mxactr = {n_max_active}\n"
        "    irivcb = {save_budget}\n"
    )


class PackageGroups(Enum):
    chd = ConstantHeadGroup
    drn = DrainageGroup
    ghb = GeneralHeadBoundaryGroup
    riv = RiverGroup
