from imod.mf6.pkgbase import Package


class TimeDiscretization(Package):
    _pkg_id = "timedis"

    def __init__(self, timestep_duration, n_timesteps=1, transient=True):
        self["timestep_duration"] = timestep_duration
        self["n_timesteps"] = n_timesteps
        self["transient"] = transient
