import jinja2

from imod.flow.pkgbase import Package


class TimeDiscretization(Package):
    """
    Time discretisation package class.

    Parameters
    ----------
    timestep_duration: float
        is the length of the current stress period (PERLEN). If the flow
        solution is transient, timestep_duration specified here must be equal to
        that specified for the flow model. If the flow solution is steady-state,
        timestep_duration can be set to any desired length.
    n_timesteps: int, optional
        is the number of time steps for the transient flow solution in the
        current stress period (NSTP). If the flow solution is steady-state,
        n_timestep=1. Default value is 1.
    transient: bool, optional
        Flag indicating wether the flow simulation is transient (True) or False
        (Steady State).
        Default is True.
    timestep_multiplier: float, optional
        is the multiplier for the length of successive time steps used in the
        transient flow solution (TSMULT); it is used only if n_timesteps>1.
        timestep_multiplier>0, the length of each flow time step within the
        current stress period is calculated using the geometric progression as
        in MODFLOW. Note that both n_timesteps and timestep_multiplier specified
        here must be identical to those specified in the flow model if the flow
        model is transient.
        timestep_multiplier ≤ 0, the length of each flow time step within the
        current stress period is read from the record TSLNGH. This option is
        needed in case the length of time steps for the flow solution is not
        based on a geometric progression in a flow model, unlike MODFLOW.
        Default is 1.0.
    """

    _pkg_id = "dis"

    def __init__(
        self,
        timestep_duration,
        n_timesteps=1,
        transient=True,
        timestep_multiplier=1.0 #TODO: Does iMODFLOW support the timestep multiplier?
    ):
        super(__class__, self).__init__()
        self.dataset["timestep_duration"] = timestep_duration
        self.dataset["n_timesteps"] = n_timesteps
        self.dataset["transient"] = transient
        self.dataset["timestep_multiplier"] = timestep_multiplier

    def _render(self, globaltimes):
        d = {}
        dicts = {}
        _dis_mapping = (
            ("perlen", "timestep_duration"),
            ("nstp", "n_timesteps"),
            ("sstr", "transient"),
            ("tsmult", "timestep_multiplier"),
        )
        d["mapping"] = _dis_mapping
        datavars = [t[1] for t in _dis_mapping]
        for varname in datavars:
            dicts[varname] = self._compose_values_time(varname, globaltimes)
            if varname == "transient":
                for k, v in dicts[varname].items():
                    if v == 1:
                        dicts[varname][k] = "tr"
                    else:
                        dicts[varname][k] = "ss"
        d["dicts"] = dicts
        d["n_periods"] = len(globaltimes)

        _dis_template = jinja2.Template(
            "\n"
            "    nper = {{n_periods}}\n"
            "    {%- for name, dictname in mapping -%}"
            "        {%- for time, value in dicts[dictname].items() %}\n"
            "    {{name}}_p{{time}} = {{value}}"
            "        {%- endfor -%}"
            "    {%- endfor -%}"
        )

        return _dis_template.render(d)

    def _pkgcheck(self, **kwargs):
        to_check = [
            "timestep_duration",
            "n_timesteps",
            "transient",
        ]

        self._check_positive(to_check)
