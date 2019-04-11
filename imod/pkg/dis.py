import jinja2

from imod.pkg.pkgbase import Package


class TimeDiscretization(Package):
    _pkg_id = "dis"

    _dis_template = jinja2.Template(
        """

        """
    )

    _btn_template = jinja2.Template(
        """

        """
    )

    def __init__(n_timesteps, transient, timestep_multiplier, transport_timestep_multiplier, transport_initial_timestep):
        self.super(__class__, self).__init__()
        self["n_timesteps"] = n_timesteps
        self["flow_transient"] = transient
        self["flow_timestep_multiplier"] = timestep_multiplier
        self["transport_timestep_multiplier"] = transport_timestep_multiplier
        self["transport_initial_timestep"] = transport_initial_timestep
    
    def _render_dis(self):
        pass
    
    def _render_btn(self):
        pass
