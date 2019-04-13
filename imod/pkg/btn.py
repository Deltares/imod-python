import jinja2

from imod.pkg.pkgbase import Package


class BasicTransport(Package):
    _pkg_id = "btn"

    _mapping = (
        ("icbund", "icbund"),
        ("dz", "thickness"),
        ("prsity", "porosity"),
        ("laycon", "layer_type"),
    )

    _template = jinja2.Template(
    "[btn]\n"
    "    thkmin = {{minimum_active_thickness}}\n"
    "    cinact = {{inactive_concentration}}\n"
    "    {%- for layer, value in starting_concentration.items() %}\n"
    "    sconc_t1_l{{layer}} = {{value}}\n"
    "    {%- endfor -%}\n"
    "    {%- for name, dictname in mapping -%}\n"
    "        {%- for layer, value in dicts[dictname].items() %}\n"
    "    {{name}}_l{{layer}} = {{value}}\n"
    "        {%- endfor -%}\n"
    "    {%- endfor -%}\n"
    )

    def __init__(
        self,
        icbund,
        starting_concentration,
        porosity=0.3,
        n_species=1,
        inactive_concentration=1.0e30,
        minimum_active_thickness=0.01,
    ):
        super(__class__, self).__init__()
        self["icbund"] = icbund
        self["starting_concentration"] = starting_concentration
        self["porosity"] = porosity
        self["n_species"] = n_species
        self["inactive_concentration"] = inactive_concentration
        self["minimum_active_thickness"] = minimum_active_thickness

    def _render_notime(self, directory, layer_type, thickness):
        d = {}
        dicts = {}
        d["mapping"] = self._mapping
        d["starting_concentration"] = self._compose_values_layer(
            "starting_concentration", directory
        )
        
        data_vars = [t[1] for t in self._mapping]
        for varname in self.data_vars.keys():
            if varname == "starting_concentration":
                continue
            if varname in data_vars:
                dicts[varname] = self._compose_values_layer(varname, directory)
            else:
                d[varname] = self[varname].values

        dicts["thickness"] = self._compose_values_layer(
            "thickness", directory, da=thickness
        )
        dicts["layer_type"] = self._compose_values_layer(
            "layer_type", directory, da=layer_type
        )
        d["dicts"] = dicts
        return self._template.render(d)

