import abc

import jinja2
import numpy as np

from imod.wq.pkgbase import BoundaryCondition


class Evapotranspiration(BoundaryCondition, abc.ABC):
    __slots__ = ("_option",)
    _pkg_id = "evt"

    _mapping = (
        ("evtr", "maximum_rate"),
        ("surf", "surface"),
        ("exdp", "extinction_depth"),
    )

    _keywords = {"save_budget": {False: 0, True: 1}}

    _template = jinja2.Template(
        "[evt]\n"
        "    nevtop = {{recharge_option}}\n"
        "    ievtcb = {{save_budget}}\n"
        "    {%- for name, dictname in mapping -%}"
        "        {%- for time, timedict in dicts[dictname].items() -%}"
        "            {%- for layer, value in timedict.items() %}\n"
        "    {{name}}_p{{time}} = {{value}}\n"
        "            {%- endfor -%}\n"
        "        {%- endfor -%}"
        "    {%- endfor -%}"
    )

    def _render(self, directory, globaltimes, nlayer, *args, **kwargs):
        d = {
            "mapping": self._mapping,
            "save_budget": self["save_budget"].values,
            "recharge_option": self._option,
        }
        self._replace_keyword(d, "save_budget")

        dicts = {}
        for _, name in self._mapping:
            dicts[name] = self._compose_values_timelayer(
                name, globaltimes, directory, nlayer=nlayer
            )
        d["dicts"] = dicts

        return self._template.render(d)

    def _pkgcheck(self, ibound=None):
        pass

    def add_timemap(
        self, surface=None, maximum_rate=None, extinction_depth=None, use_cftime=False
    ):
        varnames = ["surface", "maximum_rate", "extinction_depth"]
        values = [surface, maximum_rate, extinction_depth]
        for varname, value in zip(varnames, values):
            self._add_timemap(varname, value, use_cftime)


class EvapotranspirationTopLayer(Evapotranspiration):
    __slots__ = ()

    _option = 1

    def __init__(
        self,
        maximum_rate,
        surface,
        extinction_depth,
        concentration=0.0,
        save_budget=False,
    ):
        super(__class__, self).__init__()
        self["maximum_rate"] = maximum_rate
        self["surface"] = surface
        self["extinction_depth"] = extinction_depth
        self["concentration"] = concentration
        self["save_budget"] = save_budget

    def _set_ssm_layers(self, ibound):
        self._ssm_layers = np.array([1])


class EvapotranspirationLayers(Evapotranspiration):
    __slots__ = ()

    _option = 2

    _mapping = (
        ("evtr", "maximum_rate"),
        ("surf", "surface"),
        ("exdp", "extinction_depth"),
        ("ievt", "evapotranspiration_layer"),
    )

    def __init__(
        self,
        maximum_rate,
        surface,
        extinction_depth,
        evapotranspiration_layer,
        concentration=0.0,
        save_budget=False,
    ):
        super(__class__, self).__init__()
        self["maximum_rate"] = maximum_rate
        self["surface"] = surface
        self["extinction_depth"] = extinction_depth
        self["evapotranspiration_layer"] = evapotranspiration_layer
        self["concentration"] = concentration
        self["save_budget"] = save_budget

    def _set_ssm_layers(self, ibound):
        unique_layers = np.unique(self["recharge_layer"].values)
        unique_layers = unique_layers[~np.isnan(unique_layers)]
        self._ssm_layers = unique_layers.astype(np.int)


class EvapotranspirationHighestActive(Evapotranspiration):
    __slots__ = ()

    _option = 3

    def __init__(
        self,
        maximum_rate,
        surface,
        extinction_depth,
        concentration=0.0,
        save_budget=False,
    ):
        super(__class__, self).__init__()
        self["maximum_rate"] = maximum_rate
        self["surface"] = surface
        self["extinction_depth"] = extinction_depth
        self["concentration"] = concentration
        self["save_budget"] = save_budget

    def _set_ssm_layers(self, ibound):
        top_layer = ibound["layer"].where(ibound > 0).min("layer")
        top_layer = top_layer.where((ibound > 0).any("layer"))
        unique_layers = np.unique(top_layer.values)
        unique_layers = unique_layers[~np.isnan(unique_layers)]
        self._ssm_layers = unique_layers.astype(np.int)
