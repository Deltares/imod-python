import numpy as np

from imod.mf6.pkgbase import Package


class TimeDiscretization(Package):
    _pkg_id = "timedis"

    def __init__(self, timestep_duration, n_timesteps=1, timestep_multiplier=1.0):
        super(__class__, self).__init__()
        self["timestep_duration"] = timestep_duration
        self["n_timesteps"] = n_timesteps
        self["timestep_multiplier"] = timestep_multiplier
        self._initialize_template()

    def render(self):
        d = {}
        d["time_units"] = "days"
        timestep_duration = self["timestep_duration"]
        n_timesteps = self["n_timesteps"]
        timestep_multiplier = self["timestep_multiplier"]
        nper = timestep_duration.size  # scalar will also have size 1
        d["nper"] = nper

        # Broadcast everything to a 1D array so it's iterable
        # also fills in a scalar value everywhere
        broadcast_array = np.ones(nper)
        timestep_duration *= broadcast_array
        n_timesteps *= broadcast_array
        timestep_multiplier *= broadcast_array

        # Zip through the arrays
        period_data = []
        for (perlen, nstp, tsmult) in enumerate(
            zip(timestep_multiplier, n_timesteps, timestep_multiplier)
        ):
            period_data.append({"perlen": perlen, "nstp": nstp, "tsmult": tsmult})
        d["period_data"] = period_data

        return self._template.render(**d)

    def write(self, directory):
        timedis_content = self.render()
        timedis_path = directory / "mfsim.tdis"  # matches nicely with mfsim.nam
        with open(timedis_path, "w") as f:
            f.write(timedis_content)
