import jinja2

from imod.pkg.pkgbase import Package


class TimeDiscretization(Package):
    _pkg_id = "dis"

    _dis_mapping = (
        ("perlen", "timestep_duration"),
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
        self["transient"] = transient
        self["flow_timestep_multiplier"] = timestep_multiplier
        self["transport_timestep_multiplier"] = transport_timestep_multiplier
        self["transport_initial_timestep"] = transport_initial_timestep

    def _compose_values_time(self, key, globaltimes):
        da = self[key]
        values = {}

        if "time" in da.coords:
            # TODO: get working for cftime
            package_times = [
                pd.to_datetime(t) for t in np.atleast_1d(da.coords["time"].values)
            ]

        for timestep, globaltime in enumerate(globaltimes):
            if "time" in da.coords:
                # forward fill
                # TODO: do smart forward fill using the colon notation
                time = list(filter(lambda t: t <= globaltime, package_times))[-1]
                # Offset 0 counting in Python, add one
                values[timestep + 1] = da.isel(time=timestep).values[()]
            else:
                values["?"] = da.values[()]
        return values

    def _render_dis(self, globaltimes):
        d = {}
        datavars = [t[1] for t in self._dis_mapping]
        for varname in datavars:
            d[varname] = self._compose_values_time(varname, globaltimes)
        return self._dis_template.render(d)

    def _render_btn(self):
        d = {}
        datavars = [t[1] for t in self._dis_mapping]
        for varname in datavars:
            d[varname] = self._compose_values_time(varname, globaltimes)
        return self._btn_template.render(d)

