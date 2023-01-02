import collections
import pathlib
import subprocess
import warnings

import jinja2
import numpy as np
import xarray as xr

import imod


class Modflow6Simulation(collections.UserDict):
    def _initialize_template(self):
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        self._template = env.get_template("sim-nam.j2")

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.directory = None
        self._initialize_template()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def time_discretization(self, times):
        warnings.warn(
            f"{self.__class__.__name__}.time_discretization() is deprecated. "
            f"In the future call {self.__class__.__name__}.create_time_discretization().",
            DeprecationWarning,
        )
        self.create_time_discretization(additional_times=times)

    def create_time_discretization(self, additional_times):
        """
        Collect all unique times from model packages and additional given
        `times`. These unique times are used as stress periods in the model. All
        stress packages must have the same starting time. Function creates
        TimeDiscretization object which is set to self["time_discretization"]

        The time discretization in imod-python works as follows:

        - The datetimes of all packages you send in are always respected
        - Subsequently, the input data you use is always included fully as well
        - All times are treated as starting times for the stress: a stress is
          always applied until the next specified date
        - For this reason, a final time is required to determine the length of
          the last stress period
        - Additional times can be provided to force shorter stress periods &
          more detailed output
        - Every stress has to be defined on the first stress period (this is a
          modflow requirement)

        Or visually (every letter a date in the time axes):

        >>> recharge a - b - c - d - e - f
        >>> river    g - - - - h - - - - j
        >>> times    - - - - - - - - - - - i
        >>> model    a - b - c h d - e - f i

        with the stress periods defined between these dates. I.e. the model
        times are the set of all times you include in the model.

        Parameters
        ----------
        additional_times : str, datetime; or iterable of str, datetimes.
            Times to add to the time discretization. At least one single time
            should be given, which will be used as the ending time of the
            simulation.

        Note
        ----
        To set the other parameters of the TimeDiscretization object, you have
        to set these to the object after calling this function.

        Example
        -------
        >>> simulation = imod.mf6.Modflow6Simulation("example")
        >>> simulation.create_time_discretization(times=["2000-01-01", "2000-01-02"])
        >>> # Set number of timesteps
        >>> simulation["time_discretization"]["n_timesteps"] = 5
        """
        self.use_cftime = any(
            [model._use_cftime() for model in self.values() if model._pkg_id == "model"]
        )

        times = [
            imod.wq.timeutil.to_datetime(time, self.use_cftime)
            for time in additional_times
        ]
        for model in self.values():
            if model._pkg_id == "model":
                times.extend(model._yield_times())

        # np.unique also sorts
        times = np.unique(np.hstack(times))

        duration = imod.wq.timeutil.timestep_duration(times, self.use_cftime)
        # Generate time discretization, just rely on default arguments
        # Probably won't be used that much anyway?
        timestep_duration = xr.DataArray(
            duration, coords={"time": np.array(times)[:-1]}, dims=("time",)
        )
        self["time_discretization"] = imod.mf6.TimeDiscretization(
            timestep_duration=timestep_duration
        )

    def render(self):
        """Renders simulation namefile"""
        d = {}
        models = []
        solutiongroups = []

        for key, value in self.items():
            if value._pkg_id == "tdis":
                d["tdis6"] = f"{key}.tdis"
            elif value._pkg_id == "model":
                models.append((value._model_type, f"{key}/{key}.nam", key))
            elif value._pkg_id == "ims":
                slnnames = value["modelnames"].values
                modeltypes = set()
                for name in slnnames:
                    try:
                        modeltypes.add(type(self[name]))
                    except KeyError:
                        raise KeyError(f"model {name} of {key} not found")
                if len(modeltypes) > 1:
                    raise ValueError(
                        "Only a single type of model allowed in a solution"
                    )
                solutiongroups.append(("ims6", f"{key}.ims", slnnames))

        d["models"] = models
        if len(models) > 1:
            d["exchanges"] = self.get_exchange_relationships()

        d["solutiongroups"] = [solutiongroups]
        return self._template.render(d)

    def write(self, directory=".", binary=True):
        # Check models for required content
        for key, model in self.items():
            # skip timedis, exchanges
            if model._pkg_id == "model":
                model._model_checks(key)

        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        # Write simulation namefile
        mfsim_content = self.render()
        mfsim_path = directory / "mfsim.nam"
        with open(mfsim_path, "w") as f:
            f.write(mfsim_content)

        # Write time discretization file
        self["time_discretization"].write(directory, "time_discretization")

        # Write individual models
        globaltimes = self["time_discretization"]["time"].values
        for key, value in self.items():
            # skip timedis, exchanges
            if value._pkg_id == "model":
                value.write(
                    directory=directory,
                    modelname=key,
                    globaltimes=globaltimes,
                    binary=binary,
                )
            elif value._pkg_id == "ims":
                value.write(
                    directory=directory,
                    pkgname=key,
                    globaltimes=globaltimes,
                    binary=binary,
                )
        self.directory = directory

    def run(self, mf6path="mf6") -> None:
        if self.directory is None:
            raise RuntimeError(f"Simulation {self.name} has not been written yet.")
        with imod.util.cd(self.directory):
            result = subprocess.run(mf6path, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Simulation {self.name}: {mf6path} failed to run with returncode "
                    f"{result.returncode}, and error message:\n\n{result.stdout.decode()} "
                )

    def write_qgis_project(self, crs, directory=".", aggregate_layers=False):
        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        with imod.util.cd(directory):
            for key, value in self.items():
                # skip timedis, exchanges
                if value._pkg_id == "model":
                    value.write_qgis_project(
                        key, crs, aggregate_layers=aggregate_layers
                    )

    def get_exchange_relationships(self):
        result = []
        flowmodels = self.get_models_of_type("gwf6")
        transportmodels = self.get_models_of_type("gwt6")
        if len(flowmodels) == 1 and len(transportmodels) > 0:
            exchange_type = "GWF6-GWT6"
            modelname_a = list(flowmodels.keys())[0]
            for counter, key in enumerate(transportmodels.keys()):
                filename = f"simulation{counter}.exg"
                modelname_b = key
                result.append((exchange_type, filename, modelname_a, modelname_b))
        return result

    def get_models_of_type(self, modeltype):
        result = {}
        for key, value in self.items():
            if value._pkg_id == "model":
                if value._model_type == modeltype:
                    result[key] = value
        return result
