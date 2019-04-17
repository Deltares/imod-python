import jinja2
import numpy as np
import xarray as xr

from imod.io import util
from imod.pkg.pkgbase import Package


class LayerPropertyFlow(Package):
    _pkg_id = "lpf"

    _mapping = (
        ("laytyp", "layer_type"),
        ("layavg", "interblock"),
        ("chani", "horizontal_anisotropy"),
        ("hk", "k_horizontal"),
        ("vka", "k_vertical"),
        ("ss", "specific_storage"),
        ("sy", "specific_yield"),
        ("laywet", "layer_wet"),
    )

    _template = jinja2.Template(
    "[lpf]\n"
    "    ilpfcb = {{save_budget}}\n"
    "    hdry = {{head_dry}}\n"
    "    layvka_l? = 0\n"
    "    {%- for name, dictname in mapping -%}\n"
    "        {%- for layer, value in dicts[dictname].items() %}\n"
    "    {{name}}_l{{layer}} = {{value}}\n"
    "        {%- endfor -%}\n"
    "    {%- endfor -%}\n"
    )

    _keywords = {
        "save_budget": {False: 0, True: 1},
        "method_wet": {"wetfactor": 0, "bottom": 1},
    }

    def __init__(
        self,
        k_horizontal,
        k_vertical,
        horizontal_anisotropy,
        interblock,
        layer_type,
        specific_storage,
        specific_yield,
        save_budget,
        layer_wet,
        interval_wet,
        method_wet,
        head_dry=1.0e20,
    ):
        super(__class__, self).__init__()
        self["k_horizontal"] = k_horizontal
        self["k_vertical"] = k_vertical
        self["horizontal_anisotropy"] = horizontal_anisotropy
        self["interblock"] = interblock
        self["layer_type"] = layer_type
        self["specific_storage"] = specific_storage
        self["specific_yield"] = specific_yield
        self["save_budget"] = save_budget
        self["layer_wet"] = layer_wet
        self["interval_wet"] = interval_wet
        self["method_wet"] = method_wet
        self["head_dry"] = head_dry

    def _render(self, directory, *args, **kwargs):
        d = {}
        d["mapping"] = self._mapping
        dicts = {}

        da_vars = [t[1] for t in self._mapping]
        for varname in self.data_vars.keys():
            if varname in da_vars:
                dicts[varname] = self._compose_values_layer(varname, directory)
            else:
                d[varname] = self[varname].values
                if varname == "save_budget" or varname == "method_wet":
                    self._replace_keyword(d, varname)
        d["dicts"] = dicts

        return self._template.render(d)
