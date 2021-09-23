import jinja2

from imod.wq.pkgbase import Package


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
    max_n_transport_timestep: int, optional
        is the maximum number of transport steps allowed within one time step of
        the flow solution (mxstrn). If the number of transport steps within a
        flow time step exceeds max_n_transport_timestep, the simulation is
        terminated.
        Default is 50_000.
    transport_timestep_multiplier: float or {"None"}, optional
        is the multiplier for successive transport steps within a flow time step
        (TTSMULT).
        If the Generalized Conjugate Gradient (GCG) solver is used and the
        solution option for the advection term is the standard finite difference
        method. A value between 1.0 and 2.0 is generally adequate. If the GCG
        package is not used, the transport solution is solved explicitly as in
        the original MT3D code, and transport_timestep_multiplier is always set
        to 1.0 regardless of the user-specified input. Note that for the
        particle tracking based solution options and the 3rd-order TVD scheme,
        transport_timestep_multiplier does not apply.
        Default is {"None"}.
    transport_initial_timestep: int, optional
        is the user-specified transport stepsize within each time step of the
        flow solution (DT0).
        transport_initial_timestep is interpreted differently depending on
        whether the solution option chosen is explicit or implicit: For explicit
        solutions (i.e., the GCG solver is not used), the program will always
        calculate a maximum transport stepsize which meets the various stability
        criteria. Setting transport_initial_timestep to zero causes the model
        calculated transport stepsize to be used in the simulation. However, the
        model-calculated transport_initial_timestep may not always be optimal.
        In this situation, transport_initial_timestep should be adjusted to find
        a value that leads to the best results. If transport_initial_timestep is
        given a value greater than the model-calculated stepsize, the
        model-calculated stepsize, instead of the user-specified value, will be
        used in the simulation.
        For implicit solutions (i.e., the GCG solver is used),
        transport_initial_timestep is the initial transport stepsize. If it is
        specified as zero, the model-calculated value of
        transport_initial_timestep, based on the user-specified Courant number
        in the Advection Package, will be used. The subsequent transport
        stepsize may increase or remain constant depending on the userspecified
        transport stepsize multiplier transport_timestep_multiplier and the
        solution scheme for the advection term.
        Default is 0.
    """

    _pkg_id = "dis"

    def __init__(
        self,
        timestep_duration,
        n_timesteps=1,
        transient=True,
        timestep_multiplier=1.0,
        max_n_transport_timestep=50_000,
        transport_timestep_multiplier=None,
        transport_initial_timestep=0.0,
    ):
        super(__class__, self).__init__()
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

    def _render_btn(self, globaltimes):
        d = {}
        dicts = {}
        # TODO: check what's necessary
        _btn_mapping = (
            #        ("perlen", "duration"), # should not be necessary
            #        ("nstp", "n_timesteps"), # should not be necessary
            ("tsmult", "timestep_multiplier"),
            #        ("tslngh", "timestep_length"),
            ("dt0", "transport_initial_timestep"),
            ("ttsmult", "transport_timestep_multiplier"),
            ("mxstrn", "max_n_transport_timestep"),
        )
        _btn_template = jinja2.Template(
            "    {%- for name, dictname in mapping -%}"
            "        {%- for time, value in dicts[dictname].items() %}\n"
            "    {{name}}_p{{time}} = {{value}}"
            "        {%- endfor -%}"
            "    {%- endfor -%}"
        )
        mapping = tuple(
            [(k, v) for k, v in _btn_mapping if v in self.dataset.data_vars]
        )
        d["mapping"] = mapping
        datavars = [t[1] for t in mapping]
        for varname in datavars:
            dicts[varname] = self._compose_values_time(varname, globaltimes)
        d["dicts"] = dicts
        return _btn_template.render(d)

    def _pkgcheck(self, ibound=None):
        to_check = [
            "timestep_duration",
            "n_timesteps",
            "transient",
            "timestep_multiplier",
            "max_n_transport_timestep",
            "transport_initial_timestep",
        ]
        if "transport_timestep_multiplier" in self.dataset.data_vars:
            to_check.append("transport_timestep_multiplier")
        self._check_positive(to_check)
