import jinja2

from imod.pkg.pkgbase import Package


class BasicFlow(Package):
    _pkg_id = "bas"
    _template = jinja2.Template(
    """
    [bas6]
        {%- for layer, value in ibound.items() %}
        ibound_l{{layer}} = {{value}}
        {%- endfor -%}
        hnoflo = {{inactive_head}}
        {%- for layer, value in starting_head.items() %}
        strt_l{{layer}} = {{value}}
        {%- endfor -%}
    """
    )

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

    def __init__(
        self,
        ibound,
        top,
        bottom,
        starting_head,
        inactive_head=1.0e30,
        confining_bed_below=0
    ):
        _check_ibound(ibound)
        super(__class__, self).__init__()
        self["ibound"] = ibound
        self["top"] = top
        self["bottom"] = bottom
        self["starting_head"] = starting_head
        self["inactive_head"] = inactive_head
        self["confining_bed_below"] = confining_bed_below
        # TODO: create dx, dy if they don't exist

    def _check_ibound(ibound):
        if not isinstance(xr.DataArray):
            raise ValueError
        if not len(ibound.shape) == 3:
            raise ValueError
    
    def _render_bas(self):
        d = {}
        d["mapping"] = self._mapping

        dicts = {}
        for varname in ("ibound", "starting_head"):
            dicts[varname] = self._compose_values_layer(varname, directory)
        d["dicts"] = dicts

        return self._template.render(d)
    
    def _render_dis(self):
        d = {}

        for varname in ("top", "bottom"):
            d[varname] = self._compose_values_layer(varname, directory)

        d["nlay"], d["nrow"], d["ncol"] = self.shape 
        d["dx"] = self.coords["dx"]
        d["dy"] = self.coords["dy"]

        return self._dis_template.render(d)

