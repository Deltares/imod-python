import collections
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.package import Package
from imod.mf6.utilities.dataset import is_dataarray_none
from imod.mf6.utilities.logging_decorators import init_log_decorator
from imod.mf6.utilities.regrid import RegridderType
from imod.mf6.write_context import WriteContext
from imod.schemata import DTypeSchema

OUTPUT_EXT_MAPPING = {
    "head": "hds",
    "concentration": "ucn",
    "budget": "cbc",
}


class OutputControl(Package, IRegridPackage):
    """
    The Output Control Option determines how and when heads, budgets and/or
    concentrations are printed to the listing file and/or written to a separate
    binary output file.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.4.2.pdf#page=53

    Currently the settings "first", "last", "all", and "frequency"
    are supported, the "steps" setting is not supported, because of
    its ragged nature. Furthermore, only one setting per stress period
    can be specified in imod-python.

    Parameters
    ----------
    save_head : {string, integer}, or xr.DataArray of {string, integer}, optional
        String or integer indicating output control for head file (.hds)
        If string, should be one of ["first", "last", "all"].
        If integer, interpreted as frequency.
    save_budget : {string, integer}, or xr.DataArray of {string, integer}, optional
        String or integer indicating output control for cell budgets (.cbc)
        If string, should be one of ["first", "last", "all"].
        If integer, interpreted as frequency.
    save_concentration : {string, integer}, or xr.DataArray of {string, integer}, optional
        String or integer indicating output control for concentration file (.ucn)
        If string, should be one of ["first", "last", "all"].
        If integer, interpreted as frequency.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.

    Examples
    --------
    To specify a mix of both 'frequency' and 'first' setting,
    we need to specify an array with both integers and strings.
    For this we need to create a numpy object array first,
    otherwise xarray converts all to strings automatically.

    >>> time = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]
    >>> data = np.array(["last", 5], dtype="object")
    >>> save_head = xr.DataArray(data, coords={"time": time}, dims=("time"))
    >>> oc = imod.mf6.OutputControl(save_head=save_head, save_budget=None, save_concentration=None)

    """

    _pkg_id = "oc"
    _keyword_map = {}
    _template = Package._initialize_template(_pkg_id)

    _init_schemata = {
        "save_head": [
            DTypeSchema(np.integer) | DTypeSchema(str) | DTypeSchema(object),
        ],
        "save_budget": [
            DTypeSchema(np.integer) | DTypeSchema(str) | DTypeSchema(object),
        ],
        "save_concentration": [
            DTypeSchema(np.integer) | DTypeSchema(str) | DTypeSchema(object),
        ],
    }

    _write_schemata = {}
    _regrid_method: dict[str, Tuple[RegridderType, str]] = {}

    @init_log_decorator()
    def __init__(
        self,
        save_head=None,
        save_budget=None,
        save_concentration=None,
        head_file=None,
        budget_file=None,
        concentration_file=None,
        validate: bool = True,
    ):
        save_concentration = (
            None if is_dataarray_none(save_concentration) else save_concentration
        )
        save_head = None if is_dataarray_none(save_head) else save_head
        save_budget = None if is_dataarray_none(save_budget) else save_budget

        if save_head is not None and save_concentration is not None:
            raise ValueError("save_head and save_concentration cannot both be defined.")

        dict_dataset = {
            "save_head": save_head,
            "save_concentration": save_concentration,
            "save_budget": save_budget,
            "head_file": head_file,
            "budget_file": budget_file,
            "concentration_file": concentration_file,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def _get_ocsetting(self, setting):
        """Get oc setting based on its type. If integers return f'frequency {setting}', if"""
        if isinstance(setting, (int, np.integer)) and not isinstance(setting, bool):
            return f"frequency {setting}"
        elif isinstance(setting, str):
            if setting.lower() in ["first", "last", "all"]:
                return setting.lower()
            else:
                raise ValueError(
                    f"Output Control received wrong string. String should be one of ['first', 'last', 'all'], instead got {setting}"
                )
        else:
            raise TypeError(
                f"Output Control setting should be either integer or string in ['first', 'last', 'all'], instead got {setting}"
            )

    def _get_output_filepath(self, directory: Path, output_variable: str) -> Path:
        varname = f"{output_variable}_file"
        ext = OUTPUT_EXT_MAPPING[output_variable]
        modelname = directory.stem

        filepath = self.dataset[varname].values[()]
        if filepath is None:
            filepath = directory / f"{modelname}.{ext}"
        else:
            if not isinstance(filepath, str | Path):
                raise ValueError(
                    f"{varname} should be of type str or Path. However it is of type {type(filepath)}"
                )
            filepath = Path(filepath)

        if filepath.is_absolute():
            path = filepath
        else:
            # Get path relative to the simulation name file.
            sim_directory = directory.parent
            path = Path(os.path.relpath(filepath, sim_directory))

        return path

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}

        for output_variable in OUTPUT_EXT_MAPPING.keys():
            save = self.dataset[f"save_{output_variable}"].values[()]
            if save is not None:
                varname = f"{output_variable}_file"
                output_path = self._get_output_filepath(directory, output_variable)
                d[varname] = output_path.as_posix()

        periods = collections.defaultdict(dict)
        for datavar in ("save_head", "save_concentration", "save_budget"):
            if self.dataset[datavar].values[()] is None:
                continue
            key = datavar.replace("_", " ")
            if "time" in self.dataset[datavar].coords:
                package_times = self.dataset[datavar].coords["time"].values
                starts = np.searchsorted(globaltimes, package_times) + 1
                for i, s in enumerate(starts):
                    setting = self.dataset[datavar].isel(time=i).item()
                    periods[s][key] = self._get_ocsetting(setting)

            else:
                setting = self.dataset[datavar].item()
                periods[1][key] = self._get_ocsetting(setting)

        d["periods"] = periods

        return self._template.render(d)

    def write(
        self,
        pkgname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        write_context: WriteContext,
    ):
        # We need to overload the write here to ensure the output directory is
        # created in advance for MODFLOW6.
        super().write(pkgname, globaltimes, write_context)

        for datavar in ("head_file", "concentration_file", "budget_file"):
            path = self.dataset[datavar].values[()]
            if path is not None:
                if not isinstance(path, str):
                    raise ValueError(
                        f"{path} should be of type str. However it is of type {type(path)}"
                    )
                filepath = Path(path)
                filepath.parent.mkdir(parents=True, exist_ok=True)
        return

    @property
    def is_budget_output(self) -> bool:
        return self.dataset["save_budget"].values[()] is not None
    
    def get_regrid_methods(self) -> Optional[dict[str, Tuple[RegridderType, str]]]:
        return self._regrid_method