import collections
import pathlib
import subprocess
import warnings

import jinja2
import numpy as np
import tomli
import tomli_w
import xarray as xr

import imod
from imod.mf6.model import (
    GroundwaterFlowModel,
    GroundwaterTransportModel,
    Modflow6Model,
)
from imod.mf6.statusinfo import NestedStatusInfo
from imod.schemata import ValidationError


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

    def create_time_discretization(self, additional_times, validate: bool = True):
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
            [
                model._use_cftime()
                for model in self.values()
                if isinstance(model, Modflow6Model)
            ]
        )

        times = [
            imod.wq.timeutil.to_datetime(time, self.use_cftime)
            for time in additional_times
        ]
        for model in self.values():
            if isinstance(model, Modflow6Model):
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
            timestep_duration=timestep_duration, validate=validate
        )

    def render(self):
        """Renders simulation namefile"""
        d = {}
        models = []
        solutiongroups = []

        for key, value in self.items():
            if isinstance(value, Modflow6Model):
                models.append((value._model_id, f"{key}/{key}.nam", key))
            elif value._pkg_id == "tdis":
                d["tdis6"] = f"{key}.tdis"
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

    def write(self, directory=".", binary=True, validate: bool = True):
        """
        Write Modflow6 simulation, including assigned groundwater flow and
        transport models.

        Parameters
        ----------
        directory: str, pathlib.Path
            Directory to write Modflow 6 simulation to.
        binary: ({True, False}, optional)
            Whether to write time-dependent input for stress packages as binary
            files, which are smaller in size, or more human-readable text files.
        validate: ({True, False}, optional)
            Whether to validate the Modflow6 simulation, including models, at
            write. If True, erronous model input will throw a
            ``ValidationError``.
        """
        # Check models for required content
        for key, model in self.items():
            # skip timedis, exchanges
            if isinstance(model, Modflow6Model):
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
        status_info = NestedStatusInfo("Simulation validation status")
        globaltimes = self["time_discretization"]["time"].values
        for key, value in self.items():
            # skip timedis, exchanges
            if isinstance(value, Modflow6Model):
                status_info.add(
                    value.write(
                        directory=directory,
                        globaltimes=globaltimes,
                        modelname=key,
                        binary=binary,
                        validate=validate,
                    )
                )
            elif value._pkg_id == "ims":
                value.write(
                    directory=directory,
                    pkgname=key,
                    globaltimes=globaltimes,
                    binary=binary,
                )

        if status_info.has_errors():
            raise ValidationError("\n" + status_info.to_string())

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

    def dump(
        self, directory=".", validate: bool = True, mdal_compliant: bool = False
    ) -> None:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        toml_content = collections.defaultdict(dict)
        for key, value in self.items():
            cls_name = type(value).__name__
            if isinstance(value, Modflow6Model):
                model_toml_path = value.dump(directory, key, validate, mdal_compliant)
                toml_content[cls_name][key] = model_toml_path.relative_to(
                    directory
                ).as_posix()
            else:
                path = f"{key}.nc"
                value.dataset.to_netcdf(directory / path)
                toml_content[cls_name][key] = path

        with open(directory / f"{self.name}.toml", "wb") as f:
            tomli_w.dump(toml_content, f)

        return

    @staticmethod
    def from_file(toml_path):
        classes = {
            item_cls.__name__: item_cls
            for item_cls in (
                GroundwaterFlowModel,
                GroundwaterTransportModel,
                imod.mf6.TimeDiscretization,
                imod.mf6.Solution,
            )
        }

        toml_path = pathlib.Path(toml_path)
        with open(toml_path, "rb") as f:
            toml_content = tomli.load(f)

        simulation = Modflow6Simulation(name=toml_path.stem)
        for key, entry in toml_content.items():
            item_cls = classes[key]
            for name, filename in entry.items():
                path = toml_path.parent / filename
                simulation[name] = item_cls.from_file(path)

        return simulation

    def write_qgis_project(self, crs, directory=".", aggregate_layers=False):
        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        with imod.util.cd(directory):
            for key, value in self.items():
                # skip timedis, exchanges
                if isinstance(value, Modflow6Model):
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
        return {
            k: v
            for k, v in self.items()
            if isinstance(v, Modflow6Model) and (v._model_id == modeltype)
        }

    def clip_box(
        self,
        time_min=None,
        time_max=None,
        layer_min=None,
        layer_max=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        states_for_boundary=None,
    ) -> "Modflow6Simulation":
        """
        Clip a simulation by a bounding box (time, layer, y, x).

        Slicing intervals may be half-bounded, by providing None:

        * To select 500.0 <= x <= 1000.0:
          ``clip_box(x_min=500.0, x_max=1000.0)``.
        * To select x <= 1000.0: ``clip_box(x_min=None, x_max=1000.0)``
          or ``clip_box(x_max=1000.0)``.
        * To select x >= 500.0: ``clip_box(x_min = 500.0, x_max=None.0)``
          or ``clip_box(x_min=1000.0)``.

        Parameters
        ----------
        time_min: optional
        time_max: optional
        layer_min: optional, int
        layer_max: optional, int
        x_min: optional, float
        x_max: optional, float
        y_min: optional, float
        y_max: optional, float
        states_for_boundary : optional, Dict[pkg_name:str, boundary_values:Union[xr.DataArray, xu.UgridDataArray]]

        Returns
        -------
        clipped : Simulation
        """
        clipped = type(self)(name=self.name)
        for key, value in self.items():
            state_for_boundary = (
                None if states_for_boundary is None else states_for_boundary.get(key)
            )

            clipped[key] = value.clip_box(
                time_min=time_min,
                time_max=time_max,
                layer_min=layer_min,
                layer_max=layer_max,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                state_for_boundary=state_for_boundary,
            )
        return clipped
