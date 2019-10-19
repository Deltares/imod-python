import numpy as np

from imod.mf6.pkgbase import Package


class TimeDiscretization(Package):
    __slots__ = ("timestep_duration", "n_timesteps", "timestep_multiplier")
    _pkg_id = "tdis"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, timestep_duration, n_timesteps=1, timestep_multiplier=1.0):
        super(__class__, self).__init__()
        self["timestep_duration"] = timestep_duration
        self["n_timesteps"] = n_timesteps
        self["timestep_multiplier"] = timestep_multiplier

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
        broadcast_array = np.ones(nper, dtype=np.int)
        timestep_duration = np.atleast_1d(timestep_duration) * broadcast_array
        n_timesteps = np.atleast_1d(n_timesteps) * broadcast_array
        timestep_multiplier = np.atleast_1d(timestep_multiplier) * broadcast_array

        # Zip through the arrays
        perioddata = []
        for (perlen, nstp, tsmult) in zip(
            timestep_multiplier, n_timesteps, timestep_multiplier
        ):
            perioddata.append((perlen, nstp, tsmult))
        d["perioddata"] = perioddata

        return self._template.render(d)

    def write(self, directory, name):
        timedis_content = self.render()
        timedis_path = directory / f"{name}.tdis"
        with open(timedis_path, "w") as f:
            f.write(timedis_content)
