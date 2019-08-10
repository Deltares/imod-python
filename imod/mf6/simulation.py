import collections
import contextlib
import pathlib
import os

import jinja2
import numpy as np
import xarray as xr

import imod


@contextlib.contextmanager
def _remember_cwd():
    """
    from:
    https://stackoverflow.com/questions/169070/how-do-i-write-a-decorator-that-restores-the-cwd
    """
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)


class Modflow6Simulation(collections.UserDict):
    def _initialize_template(self):
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader)
        self._template = env.get_template("sim-nam.j2")

    def __init__(self, name):
        super(__class__, self).__init__()
        self.name = name
        self._initialize_template()

    def __setitem__(self, key, value):
        super(__class__, self).__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def time_discretization(self, endtime, starttime=None, *times):
        """
        Collect all unique times
        """
        self.use_cftime = any(
            [model._use_cftime() for model in self.values() if model._pkg_id == "model"]
        )

        times = [imod.wq.timeutil.to_datetime(time, self.use_cftime) for time in times]
        for model in self.values():
            if model._pkg_id == "model":
                times.extend(model._yield_times())

        # TODO: check that endtime is later than all other times.
        times.append(imod.wq.timeutil.to_datetime(endtime, self.use_cftime))
        if starttime is not None:
            times.append(imod.wq.timeutil.to_datetime(starttime, self.use_cftime))

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
        d["solutiongroups"] = [[("im6", f"{solvername}.ims", modelnames)]]
        return self._template.render(d)

    def write(self, directory="."):
        if isinstance(directory, str):
            directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        # Write simulation namefile
        mfsim_content = self.render()
        mfsim_path = directory / "mfsim.nam"
        with open(mfsim_path, "w") as f:
            f.write(mfsim_content)

        # Write time discretization file
        self["time_discretization"].write(directory)

        # Write individual models
        globaltimes = self["time_discretization"]["time"].values
        with _remember_cwd():
            os.chdir(directory)
            for key, value in self.items():
                # skip timedis, exchanges
                if value._pkg_id == "model":
                    value.write(key, globaltimes)
                elif value._pkg_id == "ims":
                    value.write(".", key)
