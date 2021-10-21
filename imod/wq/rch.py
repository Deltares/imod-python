import abc

import jinja2
import numpy as np

from imod.wq.pkgbase import BoundaryCondition


class Recharge(BoundaryCondition, abc.ABC):
    __slots__ = ("_option",)
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

    def _render(self, directory, globaltimes, nlayer, *args, **kwargs):
        d = {
            "mapping": self._mapping,
            "save_budget": self.dataset["save_budget"].values,
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

    def repeat_stress(self, rate=None, concentration=None, use_cftime=False):
        varnames = ["rate", "concentration"]
        values = [rate, concentration]
        for varname, value in zip(varnames, values):
            self._repeat_stress(varname, value, use_cftime)


class RechargeTopLayer(Recharge):
    """
    The Recharge package is used to simulate a specified flux distributed over
    the top of the model and specified in units of length/time (usually m/d). Within MODFLOW,
    these rates are multiplied by the horizontal area of the cells to which they
    are applied to calculate the volumetric flux rates. In this class the
    Recharge gets applied to the top grid layer (NRCHOP=1).

    Parameters
    ----------
    rate: float or xr.DataArray of floats
        is the amount of recharge.
    concentration: float or xr.DataArray of floats
        is the concentration of the recharge
    save_budget: bool, optional
        flag indicating if the budget needs to be saved.
        Default is False.
    """

    _option = 1

    def __init__(self, rate, concentration, save_budget=False):
        super(__class__, self).__init__()
        self["rate"] = rate
        self["concentration"] = concentration
        self["save_budget"] = save_budget

    def _set_ssm_layers(self, ibound):
        self._ssm_layers = np.array([1])


class RechargeLayers(Recharge):
    """
    The Recharge package is used to simulate a specified flux distributed over
    the top of the model and specified in units of length/time (usually m/d). Within MODFLOW,
    these rates are multiplied by the horizontal area of the cells to which they
    are applied to calculate the volumetric flux rates. In this class the
    Recharge gets applied to a specific, specified, layer (NRCHOP=2).

    Parameters
    ----------
    rate: float or xr.DataArray of floats
        is the amount of recharge.
    recharge_layer: int or xr.DataArray of integers
        layer number variable that defines the layer in each vertical column
        where recharge is applied
    concentration: float or xr.DataArray of floats
        is the concentration of the recharge
    save_budget: bool, optional
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

    def _set_ssm_layers(self, ibound):
        unique_layers = np.unique(self.dataset["recharge_layer"].values)
        unique_layers = unique_layers[~np.isnan(unique_layers)]
        self._ssm_layers = unique_layers.astype(np.int32)


class RechargeHighestActive(Recharge):
    """
    The Recharge package is used to simulate a specified flux distributed over
    the top of the model and specified in units of length/time (usually m/d). Within MODFLOW,
    these rates are multiplied by the horizontal area of the cells to which they
    are applied to calculate the volumetric flux rates. In this class the
    Recharge gets applied to the highest active cell in each vertical column
    (NRCHOP=3).

    Parameters
    ----------
    rate: float or xr.DataArray of floats
        is the amount of recharge.
    concentration: float or xr.DataArray of floats
        is the concentration of the recharge
    save_budget: bool, optional
        flag indicating if the budget needs to be saved.
        Default is False.
    """

    _option = 3

    def __init__(self, rate, concentration, save_budget=False):
        super(__class__, self).__init__()

        rate_scalar = np.ndim(rate) == 0
        conc_scalar = np.ndim(concentration) == 0
        if rate_scalar and (not conc_scalar):
            raise ValueError("Rate cannot be scalar if concentration is non-scalar.")

        self["rate"] = rate
        self["concentration"] = concentration
        self["save_budget"] = save_budget

    def _set_ssm_layers(self, ibound):
        rate = self.dataset["rate"]
        rate_idf = ("x" in rate.dims) and ("y" in rate.dims)
        conc_scalar = np.ndim(self.dataset["concentration"]) == 0
        if rate_idf and conc_scalar:
            rch_active = (rate != 0.0) & rate.notnull()
            if "time" in rch_active.dims:
                rch_active = rch_active.any("time")
            rch_active = rch_active & (ibound > 0)
        else:
            rch_active = ibound > 0

        top_layer = ibound["layer"].where(rch_active).min("layer")
        top_layer = top_layer.where((ibound > 0).any("layer"))
        unique_layers = np.unique(top_layer.values)
        unique_layers = unique_layers[~np.isnan(unique_layers)]
        self._ssm_layers = unique_layers.astype(np.int32)

    def repeat_stress(self, rate=None, concentration=None, use_cftime=False):
        varnames = ["rate", "concentration"]
        values = [rate, concentration]
        for varname, value in zip(varnames, values):
            self._repeat_stress(varname, value, use_cftime)
