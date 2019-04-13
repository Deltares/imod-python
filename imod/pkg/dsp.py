import jinja2

from imod.pkg.pkgbase import Package

class Dispersion(Package):
    _pkg_id = "dsp"

    _mapping = (
        ("al", "longitudinal"),
        ("trpt", "ratio_horizontal"),
        ("trpv", "ratio_vertical"),
        ("dmcoef", "diffusion_coefficient"),
    )

    _template = jinja2.Template(
        "[dsp]\n"
        "    {%- for name, dictname in mapping -%}\n"
        "        {%- for layer, value in dicts[dictname].items() %}\n"
        "    {{name}}_l{{layer}} = {{value}}\n"
        "        {%- endfor -%}\n"
        "    {%- endfor -%}\n"
    )

    def __init__(
        self, longitudinal, ratio_horizontal, ratio_vertical, diffusion_coefficient
    ):
        super(__class__, self).__init__()
        self["longitudinal"] = longitudinal
        self["ratio_horizontal"] = ratio_horizontal
        self["ratio_vertical"] = ratio_vertical
        self["diffusion_coefficient"] = diffusion_coefficient
    
    def _render(self, directory):
        d = {}
        d["mapping"] = self._mapping
        dicts = {}

        for varname in self.data_vars.keys():
            dicts[varname] = self._compose_values_layer(varname, directory)
        d["dicts"] = dicts
        
        return self._template.render(d)
