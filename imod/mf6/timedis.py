import cftime
import numpy as np

from imod.mf6.pkgbase import Package, VariableMetaData


def iso8601(datetime):
    datetype = type(datetime)
    if issubclass(datetype, np.datetime64):
        return np.datetime_as_string(datetime)
    elif issubclass(datetype, cftime.datetime):
        return datetime.isoformat()
    else:
        raise TypeError(f"Expected np.datetime64 or cftime.datetime, got {datetype}")


class TimeDiscretization(Package):
    """
    Timing for all models of the simulation is controlled by the Temporal
    Discretization (TDIS) Package.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=17

    Parameters
    ----------
    timestep_duration: float
        is the length of a stress period. (PERLEN)
    n_timesteps: int, optional
        is the number of time steps in a stress period (nstp).
        Default value: 1
    timestep_multiplier: float, optional
        is the multiplier for the length of successive time steps. The length of
        a time step is calculated by multiplying the length of the previous time
        step by timestep_multiplier (TSMULT).
        Default value: 1.0
    """

    _pkg_id = "tdis"
    _keyword_map = {}
    _template = Package._initialize_template(_pkg_id)

    _metadata_dict = {
        "timestep_duration": VariableMetaData(np.floating),
        "n_timesteps": VariableMetaData(np.integer),
        "timestep_multiplier": VariableMetaData(np.floating),
    }

    def __init__(self, timestep_duration, n_timesteps=1, timestep_multiplier=1.0):
        super().__init__()
        self.dataset["timestep_duration"] = timestep_duration
        self.dataset["n_timesteps"] = n_timesteps
        self.dataset["timestep_multiplier"] = timestep_multiplier

        self._pkgcheck()

    def render(self):
        start_date_time = iso8601(self.dataset["time"].values[0])
        d = {
            "time_units": "days",
            "start_date_time": start_date_time,
        }
        timestep_duration = self.dataset["timestep_duration"]
        n_timesteps = self.dataset["n_timesteps"]
        timestep_multiplier = self.dataset["timestep_multiplier"]
        nper = timestep_duration.size  # scalar will also have size 1
        d["nper"] = nper

        # Broadcast everything to a 1D array so it's iterable
        # also fills in a scalar value everywhere
        broadcast_array = np.ones(nper, dtype=np.int32)
        timestep_duration = np.atleast_1d(timestep_duration) * broadcast_array
        n_timesteps = np.atleast_1d(n_timesteps) * broadcast_array
        timestep_multiplier = np.atleast_1d(timestep_multiplier) * broadcast_array

        # Zip through the arrays
        perioddata = []
        for (perlen, nstp, tsmult) in zip(
            timestep_duration, n_timesteps, timestep_multiplier
        ):
            perioddata.append((perlen, nstp, tsmult))
        d["perioddata"] = perioddata

        return self._template.render(d)

    def write(self, directory, name):
        timedis_content = self.render()
        timedis_path = directory / f"{name}.tdis"
        with open(timedis_path, "w") as f:
            f.write(timedis_content)
