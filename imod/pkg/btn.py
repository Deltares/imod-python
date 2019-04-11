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
    """
    [btn]
        thkmin = {{minimum_active_thickness}}
        cinact = {{inactive_concentration}}
        {%- for layer, value in starting_concentration.items() %}
        sconc_t1_l{{layer}} = {{value}}
        {%- endfor -%}
        {%- for name, dictname in mapping -%}
            {%- for layer, value in dicts[dictname].items() %}
        {{name}}_l{{layer}} = {{value}}
            {%- endfor -%}
        {%- endfor -%}
    """
    )

    def __init__(
        icbund,
        starting_concentration,
        porosity=0.3,
        n_species=1,
        concentration_inactive=1.0e30,
        minimum_active_thickness=0.01,
    ):
        super(__class__, self).__init__()
        self["icbund"] = icbund
        self["starting_concentration"] = starting_concentration
        self["porosity"] = porosity
        self["n_species"] = n_species
        self["inactive_concentration"] = inactive_concentration
        self["minimum_active_thickness"] = minimum_active_thickness



