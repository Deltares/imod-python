import jinja2

from imod.pkg.pkgbase import BoundaryCondition


class RechargeTopLayer(BoundaryCondition):
    _pkg_id = "rch"

    def __init__(self, rate, concentration, save_budget=False):
        super(__class__, self).__init__()


class RechargeLayers(BoundaryCondition):
    _pkg_id = "rch"

    def __init__(self, rate, concentration, save_budget=False):
        super(__class__, self).__init__()


class RechargeHighestActive(BoundaryCondition):
    _pkg_id = "rch"

    _mapping = (("rech", "rate"),)

    _keywords = {"save_budget": {False: 0, True: 1}}

    _template = jinja2.Template(
        "[rch]\n"
        "    nrchop = 3\n"
        "    irchcb = {{save_budget}}\n"
        "    {%- for name, dictname in mapping -%}"
        "        {%- for time, timedict in dicts[dictname].items() -%}"
        "            {%- for layer, value in timedict.items() %}\n"
        "    {{name}}_p{{time}} = {{value}}\n"
        "            {%- endfor -%}\n"
        "        {%- endfor -%}"
        "    {%- endfor -%}"
    )

    def __init__(self, rate, concentration, save_budget=False):
        super(__class__, self).__init__()
        self["rate"] = rate
        self["concentration"] = concentration
        self["save_budget"] = save_budget

    def _render(self, directory, globaltimes, *args, **kwargs):
        d = {}
        d["mapping"] = self._mapping

        d["save_budget"] = self["save_budget"].values
        self._replace_keyword(d, "save_budget")

        dicts = {}
        dicts["rate"] = self._compose_values_timelayer("rate", globaltimes, directory)
        d["dicts"] = dicts

        return self._template.render(d)
