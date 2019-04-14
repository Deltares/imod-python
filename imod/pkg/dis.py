import jinja2

from imod.pkg.pkgbase import Package


class TimeDiscretization(Package):
    _pkg_id = "dis"

    _dis_mapping = (
        ("perlen", "duration"),
        ("nstp", "n_timesteps"),
        ("sstr",  "transient"),
    )
    _btn_mapping = (
#        ("perlen", "duration"), # should not be necessary
#        ("nstp", "n_timesteps"), # should not be necessary
        ("tsmult", "timestep_multiplier"),
        ("tslngh", "timestep_length"),
        ("dt0", "transport_initial_timestep"),
        ("mxstrn", "maximum_n_timestep"),
    )

    _dis_template = jinja2.Template(
    "    {%- for name, dictname in mapping -%}"
    "        {%- for time, value in dicts[dictname].items() -%}"
    "    {{name}}_p{{time}} = {{value}}\n"
    "        {%- endfor -%}"
    "    {%- endfor -%}"
    )

    _btn_template = jinja2.Template(
    "    {%- for name, dictname in mapping -%}"
    "        {%- for time, value in dicts[dictname].items() -%}"
    "    {{name}}_p{{time}} = {{value}}\n"
    "        {%- endfor -%}"
    "    {%- endfor -%}"
    )

    def __init__(time, n_timesteps, transient, timestep_multiplier, transport_timestep_multiplier, transport_initial_timestep):
        self.super(__class__, self).__init__()
        self["time"] = time
        self["timestep_duration"] = timestep_duration
        self["n_timesteps"] = n_timesteps
        self["flow_transient"] = transient
        self["flow_timestep_multiplier"] = timestep_multiplier
        self["transport_timestep_multiplier"] = transport_timestep_multiplier
        self["transport_initial_timestep"] = transport_initial_timestep
    
    def _render_dis(self):
        pass  # Same method
    
    def _render_btn(self):
        pass  # Same method
