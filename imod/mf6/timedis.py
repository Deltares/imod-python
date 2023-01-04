import cftime
import numpy as np

from imod.mf6.pkgbase import Package
from imod.schemata import DimsSchema, DTypeSchema


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
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "tdis"
    _keyword_map = {}
    _template = Package._initialize_template(_pkg_id)

    _init_schemata = {
        "timestep_duration": [DimsSchema("time"), DTypeSchema(np.floating)],
        "n_timesteps": [DimsSchema("time") | DimsSchema(), DTypeSchema(np.integer)],
        "timestep_multiplier": [
            DimsSchema("time") | DimsSchema(),
            DTypeSchema(np.floating),
        ],
    }

    _write_schemata = {}

    def __init__(
        self, timestep_duration, n_timesteps=1, timestep_multiplier=1.0, validate=True
    ):
        super().__init__()
        self.dataset["timestep_duration"] = timestep_duration
        self.dataset["n_timesteps"] = n_timesteps
        self.dataset["timestep_multiplier"] = timestep_multiplier

        if validate:
            self._validate_at_init()

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

    def _structured_grid_dim_check(self, da):
        if da.ndim == 0:
            return  # Scalar, no check necessary
        elif da.ndim == 1:
            if da.dims != ("time",):
                raise ValueError(
                    f"1D DataArray dims can only be ('time',). "
                    f"Instead got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )
        else:
            raise ValueError(
                f"Exceeded accepted amount of dimensions for "
                f"for {da.name} in the "
                f"{self.__class__.__name__} package. "
                f"Got {da.dims}. Can be at max ('time', )."
            )

    def write(self, directory, name):
        timedis_content = self.render()
        timedis_path = directory / f"{name}.tdis"
        with open(timedis_path, "w") as f:
            f.write(timedis_content)
