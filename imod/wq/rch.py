import jinja2

from imod.wq.pkgbase import BoundaryCondition


class RechargeTopLayer(BoundaryCondition):
    """
    The Recharge package is used to simulate a specified flux distributed over
    the top of the model and specified in units of length/time. Within MODFLOW,
    these rates are multiplied by the horizontal area of the cells to which they
    are applied to calculate the volumetric flux rates. In this class the
    Recharge gets applied to the top grid layer (NRCHOP=1).

    Parameters
    ----------
    rate: float or array of floats (xr.DataArray)
        is the amount of recharge.
    concentration: float or array of floats (xr.DataArray)
        is the concentration of the recharge
    save_budget: {True, False}, optional
        flag indicating if the budget needs to be saved.
    Default is False.
    """
    _pkg_id = "rch"

    def __init__(self, rate, concentration, save_budget=False):
        super(__class__, self).__init__()


class RechargeLayers(BoundaryCondition):
    """
    The Recharge package is used to simulate a specified flux distributed over
    the top of the model and specified in units of length/time. Within MODFLOW,
    these rates are multiplied by the horizontal area of the cells to which they
    are applied to calculate the volumetric flux rates. In this class the
    Recharge gets applied to a specific, specified, layer (NRCHOP=2).

    Parameters
    ----------
    rate: float or array of floats (xr.DataArray)
        is the amount of recharge.
    concentration: float or array of floats (xr.DataArray)
        is the concentration of the recharge
    save_budget: {True, False}, optional
        flag indicating if the budget needs to be saved.
    Default is False.
    """
    _pkg_id = "rch"

    def __init__(self, rate, concentration, save_budget=False):
        super(__class__, self).__init__()


class RechargeHighestActive(BoundaryCondition):
    """
    The Recharge package is used to simulate a specified flux distributed over
    the top of the model and specified in units of length/time. Within MODFLOW,
    these rates are multiplied by the horizontal area of the cells to which they
    are applied to calculate the volumetric flux rates. In this class the
    Recharge gets applied to the highest active cell in each vertical column
    (NRCHOP=3).

    Parameters
    ----------
    rate: float or array of floats (xr.DataArray)
        is the amount of recharge.
    concentration: float or array of floats (xr.DataArray)
        is the concentration of the recharge
    save_budget: {True, False}, optional
        flag indicating if the budget needs to be saved.
    Default is False.
    """
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
