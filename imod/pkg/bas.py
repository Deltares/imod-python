import jinja2

from imod.pkg.pkgbase import Package


class Basic(Package):
    _pkg_id = "bas"
    _template = jinja2.Template(
    """
    [bas6]
        {%- for layer, value in ibound.items() %}
        ibound_l{{layer}} = {{value}}
        {%- endfor -%}
        hnoflo = {{head_inactive}}
        {%- for layer, value in shead.items() %}
        strt_l{{layer}} = {{value}}
        {%- endfor -%}
    """

    # Non-time dependent part of dis
    # Can be inferred from ibound
    _dis_template = jinja2.Template(
    """
    [dis]
        nlay = {{nlay}}
        nrow = {{nrow}}
        ncol = {{ncol}}
        delc_r? = {{dy}}
        delr_c? = {{dx}}
        top = {{top}}
        {%- for layer, value in bot.items() %}
        botm_l{{layer}} = {{value}}
        {%- endfor %}
    """
    )

    )
    def __init__(
        self,
        ibound,
        top,
        bot,
        shead,
        head_inactive=1.0e30,
        confining_bed_below=0
    ):
        super(__class__, self).__init__()
        self["ibound"] = ibound
        self["top"] = top
        self["bot"] = bot
        self["shead"] = shead
        self["head_inactive"] = head_inactive
        self["confining_bed_below"] = confining_bed_below
        # TODO: create dx, dy if they don't exist
    
    def _render(self):
        d = {}
        d["mapping"] = self._mapping

        dicts = {}
        for varname in ("ibound", "shead"):
            dicts[varname] = self._compose_values_layer(varname, directory)
        d["dicts"] = dicts

        return self._template.render(d)
    
    def _render_dis(self):
        d = {}

        for varname in ("top", "bot"):
            d[varname] = self._compose_values_layer(varname, directory)

        d["nlay"], d["nrow"], d["ncol"] = self.shape 
        d["dx"] = self.coords["dx"]
        d["dy"] = self.coords["dy"]

        return self._dis_template.render(d)

