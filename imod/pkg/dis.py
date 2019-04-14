import jinja2
from imod.pkg.pkgbase import Package


class TimeDiscretization(Package):
    _pkg_id = "dis"

    _dis_mapping = (
        ("perlen", "timestep_duration"),
        ("nstp", "n_timesteps"),
        ("sstr", "transient"),
        ("tsmult", "timestep_multiplier")
    )

    # TODO: check what's necessary
    _btn_mapping = (
        #        ("perlen", "duration"), # should not be necessary
        #        ("nstp", "n_timesteps"), # should not be necessary
        #        ("tsmult", "timestep_multiplier"),
        #        ("tslngh", "timestep_length"),
        ("dt0", "transport_initial_timestep"),
        ("ttsmult", "transport_timestep_multiplier"),
        ("mxstrn", "max_n_transport_timestep"),
    )

    _dis_template = jinja2.Template(
        "    {%- for name, dictname in mapping -%}"
        "        {%- for time, value in dicts[dictname].items() %}\n"
        "    {{name}}_p{{time}} = {{value}}"
        "        {%- endfor -%}"
        "    {%- endfor -%}"
    )

    _btn_template = _dis_template

    def __init__(
        self,
        time,
        timestep_duration,
        n_timesteps,
        transient,
        timestep_multiplier,
        max_n_transport_timestep,
        transport_timestep_multiplier=None,
        transport_initial_timestep=0,
    ):
        super(__class__, self).__init__()
        self["time"] = time
        self["timestep_duration"] = timestep_duration
        self["n_timesteps"] = n_timesteps
        self["transient"] = transient
        self["timestep_multiplier"] = timestep_multiplier
        self["max_n_transport_timestep"] = max_n_transport_timestep
        if transport_timestep_multiplier is not None:
            self["transport_timestep_multiplier"] = transport_timestep_multiplier
        self["transport_initial_timestep"] = transport_initial_timestep

    def _render(self, globaltimes):
        d = {}
        dicts = {}
        d["mapping"] = self._dis_mapping
        datavars = [t[1] for t in self._dis_mapping]
        for varname in datavars:
            dicts[varname] = self._compose_values_time(varname, globaltimes)
            if varname == "transient":
                for k, v in dicts[varname].items():
                    if v == 1:
                        dicts[varname][k] = "TR"
                    else:
                        dicts[varname][k] = "SS"
        d["dicts"] = dicts
        return self._dis_template.render(d)

    def _render_btn(self, globaltimes):
        d = {}
        dicts = {}
        mapping = tuple([(k, v) for k, v in self._btn_mapping if v in self.data_vars])
        d["mapping"] = mapping
        datavars = [t[1] for t in mapping]
        for varname in datavars:
            dicts[varname] = self._compose_values_time(varname, globaltimes)
        d["dicts"] = dicts
        return self._btn_template.render(d)

