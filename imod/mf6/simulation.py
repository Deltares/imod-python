import collections
import pathlib

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
        self._initialize_template()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def time_discretization(self, times):
        """
        Collect all unique times from boundary conditions and insert times from argument as well.

        Function creates TimeDiscretization object which is set to self["time_discretization"]

        Parameters
        ----------
        times : xr.DataArray of datetime-likes
            times to be inserted into model time discretization.

        Note
        ----
        To set the other parameters of the TimeDiscretization object,
        you have to set these to the object after calling this function.

        Example
        -------
        >>> simulation = imod.mf6.Modflow6Simulation("example")
        >>> simulation.time_discretization(times=["2000-01-01", "2000-01-02"])
        >>> # Set number of timesteps
        >>> simulation["time_discretization"]["n_timesteps"] = 5
        """
        self.use_cftime = any(
            [model._use_cftime() for model in self.values() if model._pkg_id == "model"]
        )

        times = [imod.wq.timeutil.to_datetime(time, self.use_cftime) for time in times]
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
        solvername = None
        models = []
        modelnames = []
        for key, value in self.items():
            if value._pkg_id == "tdis":
                d["tdis6"] = f"{key}.tdis"
            elif value._pkg_id == "model":
                models.append(("gwf6", f"{key}/{key}.nam", key))
                modelnames.append(key)
            elif value._pkg_id == "ims":
                solvername = key
        d["models"] = models
        if solvername is None:
            raise ValueError("No numerical solution")
        d["solutiongroups"] = [[("ims6", f"{solvername}.ims", modelnames)]]
        return self._template.render(d)

    def write(self, directory=".", binary=True):
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
        with imod.util.cd(directory):
            for key, value in self.items():
                # skip timedis, exchanges
                if value._pkg_id == "model":
                    value.write(
                        modelname=key,
                        globaltimes=globaltimes,
                        binary=binary,
                    )
                elif value._pkg_id == "ims":
                    value.write(
                        directory=".",
                        pkgname=key,
                        globaltimes=globaltimes,
                        binary=binary,
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
