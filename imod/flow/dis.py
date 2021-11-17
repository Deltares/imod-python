import jinja2

import imod.util
from imod.flow.pkgbase import Package


class TimeDiscretization(Package):
    """
    Time discretisation package class.

    Parameters
    ----------
    timestep_duration: xr.DataArray
        is the length of the current stress period (PERLEN). If the flow
        solution is transient, timestep_duration specified here must be equal
        to that specified for the flow model. If the flow solution is
        steady-state, timestep_duration can be set to any desired length.
    n_timesteps: int, optional
        is the number of time steps for the transient flow solution in the
        current stress period (NSTP). If the flow solution is steady-state,
        n_timestep=1. Default value is 1.
    transient: bool, optional
        Flag indicating wether the flow simulation is transient (True) or False
        (Steady State).  Default is True.
    timestep_multiplier: float, optional
        is the multiplier for the length of successive time steps used in the
        transient flow solution (TSMULT); it is used only if n_timesteps>1.
        timestep_multiplier>0, the length of each flow time step within the
        current stress period is calculated using the geometric progression as
        in MODFLOW. Note that both n_timesteps and timestep_multiplier
        specified here must be identical to those specified in the flow model
        if the flow model is transient. If timestep_multiplier ≤ 0, the length
        of each flow time step within the current stress period is read from
        the record TSLNGH. This option is needed in case the length of time
        steps for the flow solution is not based on a geometric progression in
        a flow model, unlike MODFLOW.  Default is 1.0.
    """

    _pkg_id = "dis"
    _variable_order = [
        "timestep_duration",
        "n_timesteps",
        "transient",
        "timestep_multiplier",
    ]

    def __init__(
        self,
        timestep_duration,
        endtime,
        n_timesteps=1,
        transient=True,
        timestep_multiplier=1.0,
    ):
        super().__init__()
        self.dataset["timestep_duration"] = timestep_duration
        self.dataset["n_timesteps"] = n_timesteps
        self.dataset["transient"] = transient
        self.dataset["timestep_multiplier"] = timestep_multiplier
        self.endtime = endtime

    def _render(self):
        """Render iMOD TIM file, which is the time discretization of the iMODFLOW model"""
        _template = jinja2.Template(
            "{% for time in timestrings%}"
            "{{time}},1,{{n_timesteps}},{{timestep_multiplier}}\n"
            "{% endfor %}\n"
        )
        times = self.dataset["time"].values
        timestrings = [imod.util._compose_timestring(time) for time in times]
        timestrings.append(imod.util._compose_timestring(self.endtime))

        d = dict(
            timestrings=timestrings,
            n_timesteps=self.dataset["n_timesteps"].item(),
            timestep_multiplier=self.dataset["timestep_multiplier"].item(),
        )

        return _template.render(**d)

    def save(self, path):
        tim_content = self._render()

        with open(path, "w") as f:
            f.write(tim_content)

    def _pkgcheck(self, **kwargs):
        to_check = [
            "timestep_duration",
            "n_timesteps",
        ]

        self._check_positive(to_check)
