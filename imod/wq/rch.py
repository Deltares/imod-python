import jinja2

from imod.wq.pkgbase import BoundaryCondition


class Recharge(BoundaryCondition):
    _pkg_id = "rch"

    _mapping = (("rech", "rate"),)

    _keywords = {"save_budget": {False: 0, True: 1}}

    _template = jinja2.Template(
        "[rch]\n"
        "    nrchop = {{recharge_option}}\n"
        "    irchcb = {{save_budget}}\n"
        "    {%- for name, dictname in mapping -%}"
        "        {%- for time, timedict in dicts[dictname].items() -%}"
        "            {%- for layer, value in timedict.items() %}\n"
        "    {{name}}_p{{time}} = {{value}}\n"
        "            {%- endfor -%}\n"
        "        {%- endfor -%}"
        "    {%- endfor -%}"
    )

    def _render(self, directory, globaltimes, *args, **kwargs):
        d = {
            "mapping": self._mapping,
            "save_budget": self["save_budget"].values,
            "recharge_option": self._option,
        }
        self._replace_keyword(d, "save_budget")

        dicts = {}
        for _, name in self._mapping:
            dicts[name] = self._compose_values_timelayer(name, globaltimes, directory)
        d["dicts"] = dicts

        return self._template.render(d)

    def _pkgcheck(self, ibound=None):
        pass


class RechargeTopLayer(Recharge):
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

    _option = 1

    def __init__(self, rate, concentration, save_budget=False):
        super(__class__, self).__init__()
        self["rate"] = rate
        self["concentration"] = concentration
        self["save_budget"] = save_budget


class RechargeLayers(Recharge):
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
    recharge_layer: float or array of integers (xr.DataArray)
        layer number variable that defines the layer in each vertical column
        where recharge is applied
    concentration: float or array of floats (xr.DataArray)
        is the concentration of the recharge
    save_budget: {True, False}, optional
        flag indicating if the budget needs to be saved.
        Default is False.
    """

    _option = 2

    _mapping = (("rech", "rate"), ("irch", "recharge_layer"))

    def __init__(self, rate, recharge_layer, concentration, save_budget=False):
        super(__class__, self).__init__()
        self["rate"] = rate
        self["recharge_layer"] = recharge_layer
        self["concentration"] = concentration
        self["save_budget"] = save_budget


class RechargeHighestActive(Recharge):
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

    _option = 3

    def __init__(self, rate, concentration, save_budget=False):
        super(__class__, self).__init__()
        self["rate"] = rate
        self["concentration"] = concentration
        self["save_budget"] = save_budget
