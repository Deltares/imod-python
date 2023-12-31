from __future__ import annotations

import collections
import copy
import pathlib
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jinja2
import numpy as np
import tomli
import tomli_w
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.gwfgwf import GWFGWF
from imod.mf6.model import (
    GroundwaterFlowModel,
    GroundwaterTransportModel,
    Modflow6Model,
)
from imod.mf6.multimodel.exchange_creator_structured import ExchangeCreator_Structured
from imod.mf6.multimodel.exchange_creator_unstructured import (
    ExchangeCreator_Unstructured,
)
from imod.mf6.multimodel.modelsplitter import create_partition_info, slice_model
from imod.mf6.out import open_cbc, open_conc, open_hds
from imod.mf6.package import Package
from imod.mf6.pkgbase import PackageBase
from imod.mf6.statusinfo import NestedStatusInfo
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.typing import GridDataArray, GridDataset
from imod.typing.grid import concat, is_unstructured, merge, merge_partitions

OUTPUT_FUNC_MAPPING = {
    "head": (open_hds, GroundwaterFlowModel),
    "concentration": (open_conc, GroundwaterTransportModel),
    "budget-flow": (open_cbc, GroundwaterFlowModel),
    "budget-transport": (open_cbc, GroundwaterTransportModel),
}


def get_models(simulation: Modflow6Simulation) -> Dict[str, Modflow6Model]:
    return {k: v for k, v in simulation.items() if isinstance(v, Modflow6Model)}


def get_packages(simulation: Modflow6Simulation) -> Dict[str, Package]:
    return {
        pkg_name: pkg
        for pkg_name, pkg in simulation.items()
        if isinstance(pkg, Package)
    }


def is_split(simulation: Modflow6Simulation) -> bool:
    return "split_exchanges" in simulation.keys()


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

    def render(self, write_context: WriteContext):
        """Renders simulation namefile"""
        d: Dict[str, Any] = {}
        models = []
        solutiongroups = []
        for key, value in self.items():
            if isinstance(value, Modflow6Model):
                model_name_file = pathlib.Path(
                    write_context.root_directory / pathlib.Path(f"{key}", f"{key}.nam")
                ).as_posix()
                models.append((value.model_id(), model_name_file, key))
            elif isinstance(value, PackageBase):
                if value._pkg_id == "tdis":
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

    def write(
        self,
        directory=".",
        binary=True,
        validate: bool = True,
        use_absolute_paths=False,
    ):
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
        absolute_paths: ({True, False}, optional)
            True if all paths written to the mf6 inputfiles should be absolute.
        """
        # create write context
        write_context = WriteContext(directory, binary, use_absolute_paths)
        if is_split(self):
            write_context.is_partitioned = True

        # Check models for required content
        for key, model in self.items():
            # skip timedis, exchanges
            if isinstance(model, Modflow6Model):
                model._model_checks(key)

        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        # Write simulation namefile
        mfsim_content = self.render(write_context)
        mfsim_path = directory / "mfsim.nam"
        with open(mfsim_path, "w") as f:
            f.write(mfsim_content)

        # Write time discretization file
        self["time_discretization"].write(directory, "time_discretization")

        # Write individual models
        status_info = NestedStatusInfo("Simulation validation status")
        globaltimes = self["time_discretization"]["time"].values
        for key, value in self.items():
            model_write_context = write_context.copy_with_new_write_directory(
                write_context.simulation_directory
            )
            # skip timedis, exchanges
            if isinstance(value, Modflow6Model):
                status_info.add(
                    value.write(
                        modelname=key,
                        globaltimes=globaltimes,
                        validate=validate,
                        write_context=model_write_context,
                    )
                )
            elif isinstance(value, PackageBase):
                if value._pkg_id == "ims":
                    ims_write_context = write_context.copy_with_new_write_directory(
                        write_context.simulation_directory
                    )
                    value.write(key, globaltimes, ims_write_context)
            elif isinstance(value, list):
                for exchange in value:
                    if isinstance(exchange, imod.mf6.GWFGWF):
                        exchange.write(
                            exchange.packagename(), globaltimes, write_context
                        )

        if status_info.has_errors():
            raise ValidationError("\n" + status_info.to_string())

        self.directory = directory

    def run(self, mf6path: Union[str, Path] = "mf6") -> None:
        """
        Run Modflow 6 simulation. This method runs a subprocess calling
        ``mf6path``. This argument is set to ``mf6``, which means the Modflow 6
        executable is expected to be added to your PATH environment variable.
        :doc:`See this writeup how to add Modflow 6 to your PATH on Windows </examples/mf6/index>`

        Note that the ``write`` method needs to be called before this method is
        called.

        Parameters
        ----------
        mf6path: Union[str, Path]
            Path to the Modflow 6 executable. Defaults to calling ``mf6``.

        Examples
        --------
        Make sure you write your model first

        >>> simulation.write(path/to/model)
        >>> simulation.run()
        """
        if self.directory is None:
            raise RuntimeError(f"Simulation {self.name} has not been written yet.")
        with imod.util.cd(self.directory):
            result = subprocess.run(mf6path, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Simulation {self.name}: {mf6path} failed to run with returncode "
                    f"{result.returncode}, and error message:\n\n{result.stdout.decode()} "
                )

    def open_head(self, dry_nan: bool = False) -> GridDataArray:
        """
        Open heads of finished simulation, requires that the ``run`` method has
        been called.

        The data is lazily read per timestep and automatically converted into
        (dense) xr.DataArrays or xu.UgridDataArrays, for DIS and DISV
        respectively. The conversion is done via the information stored in the
        Binary Grid file (GRB).

        Parameters
        ----------
        dry_nan: bool, default value: False.
            Whether to convert dry values to NaN.

        Returns
        -------
        head: Union[xr.DataArray, xu.UgridDataArray]

        Examples
        --------
        Make sure you write and run your model first

        >>> simulation.write(path/to/model)
        >>> simulation.run()

        Then open heads:

        >>> head = simulation.open_head()
        """
        return self._open_output("head", dry_nan=dry_nan)

    def open_transport_budget(
        self, species_ls: list[str] = None
    ) -> dict[str, GridDataArray]:
        """
        Open transport budgets of finished simulation, requires that the ``run``
        method has been called.

        The data is lazily read per timestep and automatically converted into
        (dense) xr.DataArrays or xu.UgridDataArrays, for DIS and DISV
        respectively. The conversion is done via the information stored in the
        Binary Grid file (GRB).

        Parameters
        ----------
        species_ls: list of strings, default value: None.
            List of species names, which will be used to concatenate the
            concentrations along the ``"species"`` dimension, in case the
            simulation has multiple species and thus multiple transport models.
            If None, transport model names will be used as species names.

        Returns
        -------
        budget: Dict[str, xr.DataArray|xu.UgridDataArray]
            DataArray contains float64 data of the budgets, with dimensions ("time",
            "layer", "y", "x").

        """
        return self._open_output("budget-transport", species_ls=species_ls)

    def open_flow_budget(self, flowja: bool = False) -> dict[str, GridDataArray]:
        """
        Open flow budgets of finished simulation, requires that the ``run``
        method has been called.

        The data is lazily read per timestep and automatically converted into
        (dense) xr.DataArrays or xu.UgridDataArrays, for DIS and DISV
        respectively. The conversion is done via the information stored in the
        Binary Grid file (GRB).

        The ``flowja`` argument controls whether the flow-ja-face array (if
        present) is returned in grid form as "as is". By default
        ``flowja=False`` and the array is returned in "grid form", meaning:

            * DIS: in right, front, and lower face flow. All flows are placed in
            the cell.
            * DISV: in horizontal and lower face flow.the horizontal flows are
            placed on the edges and the lower face flow is placed on the faces.

        When ``flowja=True``, the flow-ja-face array is returned as it is found in
        the CBC file, with a flow for every cell to cell connection. Additionally,
        a ``connectivity`` DataArray is returned describing for every cell (n) its
        connected cells (m).

        Parameters
        ----------
        flowja: bool, default value: False
            Whether to return the flow-ja-face values "as is" (``True``) or in a
            grid form (``False``).

        Returns
        -------
        budget: Dict[str, xr.DataArray|xu.UgridDataArray]
            DataArray contains float64 data of the budgets, with dimensions ("time",
            "layer", "y", "x").

        Examples
        --------
        Make sure you write and run your model first

        >>> simulation.write(path/to/model)
        >>> simulation.run()

        Then open budgets:

        >>> budget = simulation.open_flow_budget()

        Check the contents:

        >>> print(budget.keys())

        Get the drainage budget, compute a time mean for the first layer:

        >>> drn_budget = budget["drn]
        >>> mean = drn_budget.sel(layer=1).mean("time")
        """
        return self._open_output("budget-flow", flowja=flowja)

    def open_concentration(
        self, species_ls: list[str] = None, dry_nan: bool = False
    ) -> GridDataArray:
        """
        Open concentration of finished simulation, requires that the ``run``
        method has been called.

        The data is lazily read per timestep and automatically converted into
        (dense) xr.DataArrays or xu.UgridDataArrays, for DIS and DISV
        respectively. The conversion is done via the information stored in the
        Binary Grid file (GRB).

        Parameters
        ----------
        species_ls: list of strings, default value: None.
            List of species names, which will be used to concatenate the
            concentrations along the ``"species"`` dimension, in case the
            simulation has multiple species and thus multiple transport models.
            If None, transport model names will be used as species names.
        dry_nan: bool, default value: False.
            Whether to convert dry values to NaN.

        Returns
        -------
        concentration: Union[xr.DataArray, xu.UgridDataArray]

        Examples
        --------
        Make sure you write and run your model first

        >>> simulation.write(path/to/model)
        >>> simulation.run()

        Then open concentrations:

        >>> concentration = simulation.open_concentration()
        """
        return self._open_output(
            "concentration", species_ls=species_ls, dry_nan=dry_nan
        )

    def _open_output(self, output: str, **settings) -> GridDataArray | GridDataset:
        """
        Opens output of one or multiple models.

        Parameters
        ----------
        output: str
            Output variable name to open
        **settings:
            Extra settings that need to be passed through to the respective
            output function.
        """
        _, modeltype = OUTPUT_FUNC_MAPPING[output]
        modelnames = self.get_models_of_type(modeltype._model_id).keys()
        # Pop species_ls, set to modelnames in case not found
        species_ls = settings.pop("species_ls", modelnames)
        if len(modelnames) == 0:
            raise ValueError(
                f"Could not find any models of appropriate type for {output}, "
                f"make sure a model of type {modeltype} is assigned to simulation."
            )
        elif len(modelnames) == 1:
            modelname = next(iter(modelnames))
            return self._open_output_single_model(modelname, output, **settings)
        elif is_split(self):
            if "budget" in output:
                return self._merge_fluxes(modelnames, output, **settings)
            else:
                return self._merge_states(modelnames, output, **settings)
        elif output == "concentration":
            return self._concat_concentrations(
                modelnames, species_ls, output, **settings
            )
        elif output == "budget-transport":
            return self._concat_transport_budgets(
                modelnames, species_ls, output, **settings
            )
        else:
            raise RuntimeError(
                f"Unexpected error when opening {output} for {modelnames}"
            )

    def _merge_states(
        self, modelnames: list[str], output: str, **settings
    ) -> GridDataArray:
        state_partitions = []
        for modelname in modelnames:
            state_partitions.append(
                self._open_output_single_model(modelname, output, **settings)
            )
        return merge_partitions(state_partitions)

    def _merge_and_assign_exchange_fluxes(self, cbc: GridDataset) -> GridDataset:
        """
        Merge and assign exchange fluxes to cell by cell budgets:
        cbc[[gwf-gwf_1, gwf-gwf_3]] to cbc[gwf-gwf]
        """
        exchange_names = [
            key for key in cbc.keys() if ("gwf-gwf" in key) or ("gwt-gwt" in key)
        ]
        exchange_flux = cbc[exchange_names].to_array().sum(dim="variable")
        cbc = cbc.drop_vars(exchange_names)
        # "gwf-gwf" or "gwt-gwt"
        exchange_key = exchange_names[0].split("_")[0]
        cbc[exchange_key] = exchange_flux
        return cbc

    def _merge_fluxes(
        self, modelnames: list[str], output: str, **settings
    ) -> GridDataset:
        if settings["flowja"] is True:
            raise ValueError("``flowja`` cannot be set to True when merging fluxes.")

        cbc_per_partition = []
        for modelname in modelnames:
            partition_model = self[modelname]
            partition_domain = partition_model.domain
            cbc_dict = self._open_output_single_model(modelname, output, **settings)
            # Force list of dicts to list of DataArrays to work around:
            # https://github.com/Deltares/xugrid/issues/179
            cbc_list = [da.rename(key) for key, da in cbc_dict.items()]
            cbc = merge(cbc_list)
            # Merge and assign exchange fluxes to dataset
            # FUTURE: Refactor to insert these exchange fluxes in horizontal
            # flows.
            cbc = self._merge_and_assign_exchange_fluxes(cbc)
            if not is_unstructured(cbc):
                cbc = cbc.where(partition_domain, other=np.nan)
            cbc_per_partition.append(cbc)

        return merge_partitions(cbc_per_partition)

    def _concat_concentrations(
        self, modelnames: list[str], species_ls: list[str], output: str, **settings
    ) -> GridDataArray:
        concentrations = []
        for modelname, species in zip(modelnames, species_ls):
            conc = self._open_output_single_model(modelname, output, **settings)
            conc = conc.assign_coords(species=species)
            concentrations.append(conc)
        return concat(concentrations, dim="species")

    def _concat_transport_budgets(
        self, modelnames: list[str], species_ls: list[str], output: str, **settings
    ) -> GridDataset:
        budgets = []
        for modelname, species in zip(modelnames, species_ls):
            budget_dict = self._open_output_single_model(modelname, output, **settings)
            # Force list of dicts to list of DataArrays to work around:
            # https://github.com/Deltares/xugrid/issues/179
            budget_list = [da.rename(key) for key, da in budget_dict.items()]
            budget = merge(budget_list)
            budget = budget.assign_coords(species=species)
            budgets.append(budget)

        return concat(budgets, dim="species")

    def _open_output_single_model(
        self, modelname: str, output: str, **settings
    ) -> GridDataArray | dict[str, GridDataArray]:
        """
        Opens output of single model

        Parameters
        ----------
        modelname: str
            Name of groundwater model from which output should be read.
        output: str
            Output variable name to open.
        **settings:
            Extra settings that need to be passed through to the respective
            output function.
        """
        open_func, expected_modeltype = OUTPUT_FUNC_MAPPING[output]

        if self.directory is None:
            raise RuntimeError(f"Simulation {self.name} has not been written yet.")
        model_path = self.directory / modelname

        # Get model
        model = self[modelname]
        if not isinstance(model, expected_modeltype):
            raise TypeError(
                f"{modelname} not a {expected_modeltype}, instead got {type(model)}"
            )
        # Get output file path
        oc_key = model._get_pkgkey("oc")
        oc_pkg = model[oc_key]
        # Ensure "-transport" and "-flow" are stripped from "budget"
        oc_output = output.split("-")[0]
        output_path = oc_pkg._get_output_filepath(model_path, oc_output)
        # Force path to always include simulation directory.
        output_path = self.directory / output_path

        grb_path = self._get_grb_path(modelname)

        if not output_path.exists():
            raise RuntimeError(
                f"Could not find output in {output_path}, check if you already ran simulation {self.name}"
            )

        return open_func(output_path, grb_path, **settings)

    def _get_flow_modelname_coupled_to_transport_model(
        self, transport_modelname: str
    ) -> str:
        """
        Get name of flow model coupled to transport model, throws error if
        multiple flow models are couple to 1 transport model.
        """
        exchanges = self.get_exchange_relationships()
        coupled_flow_models = [
            i[2]
            for i in exchanges
            if (i[3] == transport_modelname) & (i[0] == "GWF6-GWT6")
        ]
        if len(coupled_flow_models) != 1:
            raise ValueError(
                f"Exactly one flow model must be coupled to transport model {transport_modelname}, got: {coupled_flow_models}"
            )
        return coupled_flow_models[0]

    def _get_grb_path(self, modelname: str) -> Path:
        """
        Finds appropriate grb path belonging to modelname. Grb files are not
        written for transport models, so this method always returns a path to a
        flowmodel. In case of a transport model, it returns the path to the grb
        file its coupled flow model.
        """
        model = self[modelname]
        # Get grb path
        if isinstance(model, GroundwaterTransportModel):
            flow_model_name = self._get_flow_modelname_coupled_to_transport_model(
                modelname
            )
            flow_model_path = self.directory / flow_model_name
        else:
            flow_model_path = self.directory / modelname

        diskey = model._get_diskey()
        dis_id = model[diskey]._pkg_id
        return flow_model_path / f"{diskey}.{dis_id}.grb"

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

    def get_exchange_relationships(self):
        result = []
        flowmodels = self.get_models_of_type("gwf6")
        transportmodels = self.get_models_of_type("gwt6")
        # exchange for flow and transport
        if len(flowmodels) == 1 and len(transportmodels) > 0:
            exchange_type = "GWF6-GWT6"
            modelname_a = list(flowmodels.keys())[0]
            for counter, key in enumerate(transportmodels.keys()):
                filename = f"simulation{counter}.exg"
                modelname_b = key
                result.append((exchange_type, filename, modelname_a, modelname_b))

        # exchange for splitting models
        if is_split(self):
            for exchange in self["split_exchanges"]:
                result.append(exchange.get_specification())

        return result

    def get_models_of_type(self, modeltype):
        return {
            k: v
            for k, v in self.items()
            if isinstance(v, Modflow6Model) and (v.model_id() == modeltype)
        }

    def clip_box(
        self,
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        layer_min: Optional[int] = None,
        layer_max: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        states_for_boundary: Optional[dict[str, GridDataArray]] = None,
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

        if is_split(self):
            raise RuntimeError(
                "Unable to clip simulation. Clipping can only be done on simulations that haven't been split."
            )

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

    def split(self, submodel_labels: xr.DataArray) -> Modflow6Simulation:
        """
        Split a simulation in different partitions using a submodel_labels array.

        The submodel_labels array defines how a simulation will be split. The array should have the same topology as
        the domain being split i.e. similar shape as a layer in the domain. The values in the array indicate to
        which partition a cell belongs. The values should be zero or greater.

        The method return a new simulation containing all the split models and packages
        """
        if is_split(self):
            raise RuntimeError(
                "Unable to split simulation. Splitting can only be done on simulations that haven't been split."
            )

        original_models = get_models(self)
        original_packages = get_packages(self)

        partition_info = create_partition_info(submodel_labels)

        if is_unstructured(submodel_labels):
            exchange_creator = ExchangeCreator_Unstructured(
                submodel_labels, partition_info
            )
        else:
            exchange_creator = ExchangeCreator_Structured(
                submodel_labels, partition_info
            )

        new_simulation = imod.mf6.Modflow6Simulation(f"{self.name}_partioned")
        for package_name, package in {**original_packages}.items():
            new_simulation[package_name] = package

        for model_name, model in original_models.items():
            for submodel_partition_info in partition_info:
                new_model_name = f"{model_name}_{submodel_partition_info.id}"
                new_simulation[new_model_name] = slice_model(
                    submodel_partition_info, model
                )

        exchanges = []
        for model_name, model in original_models.items():
            exchanges += exchange_creator.create_exchanges(
                model_name, model.domain.layer
            )

        new_simulation["solver"]["modelnames"] = xr.DataArray(
            list(get_models(new_simulation).keys())
        )

        new_simulation._add_modelsplit_exchanges(exchanges)
        new_simulation._set_exchange_options()

        new_simulation._filter_inactive_cells_from_exchanges()
        return new_simulation

    def regrid_like(
        self,
        regridded_simulation_name: str,
        target_grid: Union[xr.DataArray, xu.UgridDataArray],
        validate: bool = True,
    ) -> "Modflow6Simulation":
        """
        This method creates a new simulation object. The models contained in the new simulation are regridded versions
        of the models in the input object (this).
        Time discretization and solver settings are copied.

        Parameters
        ----------
        regridded_simulation_name: str
            name given to the output simulation
        target_grid: xr.DataArray or  xu.UgridDataArray
            discretization onto which the models  in this simulation will be regridded
        validate: bool
            set to true to validate the regridded packages

        Returns
        -------
        a new simulation object with regridded models
        """

        if is_split(self):
            raise RuntimeError(
                "Unable to regrid simulation. Regridding can only be done on simulations that haven't been split."
            )

        result = self.__class__(regridded_simulation_name)
        for key, item in self.items():
            if isinstance(item, GroundwaterFlowModel):
                result[key] = item.regrid_like(target_grid, validate)
            elif isinstance(item, imod.mf6.Solution) or isinstance(
                item, imod.mf6.TimeDiscretization
            ):
                result[key] = copy.deepcopy(item)
            else:
                raise NotImplementedError(f"regridding not supported for {key}")

        return result

    def _add_modelsplit_exchanges(self, exchanges_list: List[GWFGWF]) -> None:
        if not is_split(self):
            self["split_exchanges"] = []
        self["split_exchanges"].extend(exchanges_list)

    def _set_exchange_options(self):
        # collect some options that we will auto-set
        for exchange in self["split_exchanges"]:
            model_name_1 = exchange.dataset["model_name_1"].values[()]
            model_1 = self[model_name_1]
            exchange.set_options(
                save_flows=model_1["oc"].is_budget_output,
                dewatered=model_1["npf"].is_dewatered,
                variablecv=model_1["npf"].is_variable_vertical_conductance,
                xt3d=model_1["npf"].get_xt3d_option(),
                newton=model_1.is_use_newton(),
            )

    def _filter_inactive_cells_from_exchanges(self) -> None:
        for ex in self["split_exchanges"]:
            for i in [1, 2]:
                self._filter_inactive_cells_exchange_domain(ex, i)

    def _filter_inactive_cells_exchange_domain(self, ex: GWFGWF, i: int) -> None:
        """Filters inactive cells from one exchange domain inplace"""
        modelname = ex[f"model_name_{i}"].values[()]
        domain = self[modelname].domain

        layer = ex.dataset["layer"] - 1
        id = ex.dataset[f"cell_id{i}"] - 1
        if is_unstructured(domain):
            exchange_cells = {
                "layer": layer,
                "mesh2d_nFaces": id,
            }
        else:
            exchange_cells = {
                "layer": layer,
                "y": id.sel({f"cell_dims{i}": f"row_{i}"}),
                "x": id.sel({f"cell_dims{i}": f"column_{i}"}),
            }
        exchange_domain = domain.isel(exchange_cells)
        active_exchange_domain = exchange_domain.where(exchange_domain.values > 0)
        active_exchange_domain = active_exchange_domain.dropna("index")
        ex.dataset = ex.dataset.sel(index=active_exchange_domain["index"])

    def __repr__(self) -> str:
        typename = type(self).__name__
        INDENT = "    "
        attrs = [
            f"{typename}(",
            f"{INDENT}name={repr(self.name)},",
            f"{INDENT}directory={repr(self.directory)}",
        ]
        items = [
            f"{INDENT}{repr(key)}: {type(value).__name__},"
            for key, value in self.items()
        ]
        # Place the emtpy dict on the same line. Looks silly otherwise.
        if items:
            content = attrs + ["){"] + items + ["}"]
        else:
            content = attrs + ["){}"]
        return "\n".join(content)
