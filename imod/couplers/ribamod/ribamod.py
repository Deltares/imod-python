from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tomli_w
import imod
import xarray as xr

from imod.mf6 import Modflow6Simulation, River, Drainage
from imod.util import MissingOptionalModule

try:
    import geopandas as gpd
except ImportError:
    gpd = MissingOptionalModule("geopandas")

try:
    import ribasim
except ImportError:
    ribasim = MissingOptionalModule("ribasim")


@dataclass
class DriverCoupling:
    mf6_model: str
    mf6_active_river_packages: Tuple[str] = ()
    mf6_passive_river_packages: Tuple[str] = ()
    mf6_active_drainage_packages: Tuple[str] = ()
    mf6_passive_drainage_packages: Tuple[str] = ()


class RibaMod:
    """
    The RibaMod class creates the necessary input files for coupling Ribasim to
    MODFLOW 6.

    Parameters
    ----------
    ribasim_model : ribasim.model
        The Ribasim model that should be coupled.
    mf6_simulation : Modflow6Simulation
        The Modflow6 simulation that should be coupled.
    coupling_list: list of DriverCoupling
        One entry per MODFLOW 6 model that should be coupled
    """

    _toml_name = "imod_coupler.toml"
    _ribasim_model_dir = "ribasim"
    _modflow6_model_dir = "modflow6"

    def __init__(
        self,
        ribasim_model: "ribasim.Model",
        mf6_simulation: Modflow6Simulation,
        coupling_list: List[DriverCoupling],
        basin_definition: gpd.GeoDataFrame,
    ):
        self.ribasim_model = ribasim_model
        self.mf6_simulation = mf6_simulation
        self.coupling_list = coupling_list
        if "basin_id" not in basin_definition.columns:
            raise ValueError('Basin definition must contain "basin_id" column')
        self.basin_definition = basin_definition

    def write(
        self,
        directory: Union[str, Path],
        modflow6_dll: Union[str, Path],
        ribasim_dll: Union[str, Path],
        ribasim_dll_dependency: Union[str, Path],
        modflow6_write_kwargs: Optional[dict] = None,
    ):
        """
        Write Ribasim and Modflow 6 model with exchange files, as well as a
        ``.toml`` file which configures the iMOD Coupler run.

        Parameters
        ----------
        directory: str or Path
            Directory in which to write the coupled models
        modflow6_dll: str or Path
            Path to modflow6 .dll. You can obtain this library by downloading
            `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        ribasim_dll: str or Path
            Path to ribasim .dll.
        ribasim_dll_dependency: str or Path
            Directory with ribasim .dll dependencies.
        modflow6_write_kwargs: dict
            Optional dictionary with keyword arguments for the writing of
            Modflow6 models. You can use this for example to turn off the
            validation at writing (``validation=False``) or to write text files
            (``binary=False``)
        """

        if modflow6_write_kwargs is None:
            modflow6_write_kwargs = {}

        # force to Path
        directory = Path(directory)
        self.mf6_simulation.write(
            directory / self._modflow6_model_dir,
            **modflow6_write_kwargs,
        )
        self.ribasim_model.write(directory / self._ribasim_model_dir)
        coupling_dict = self.write_exchanges(directory)
        self.write_toml(
            directory, coupling_dict, modflow6_dll, ribasim_dll, ribasim_dll_dependency
        )

    def write_toml(
        self,
        directory: Union[str, Path],
        coupling_dict: Dict[str, Any],
        modflow6_dll: Union[str, Path],
        ribasim_dll: Union[str, Path],
        ribasim_dll_dependency: Union[str, Path],
    ):
        """
        Write .toml file which configures the imod coupler run.

        Parameters
        ----------
        directory: str or Path
            Directory in which to write the .toml file.
        modflow6_dll: str or Path
            Path to modflow6 .dll. You can obtain this library by downloading
            `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        ribasim_dll: str or Path
            Path to ribasim .dll.
        ribasim_dll_dependency: str or Path
            Directory with ribasim .dll dependencies.
        """
        # force to Path
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        toml_path = directory / self._toml_name
        coupler_toml = {
            "timing": False,
            "log_level": "INFO",
            "driver_type": "ribamod",
            "driver": {
                "kernels": {
                    "modflow6": {
                        "dll": str(modflow6_dll),
                        "work_dir": f".\\{self._modflow6_model_dir}",
                    },
                    "ribasim": {
                        "dll": str(ribasim_dll),
                        "dll_dep_dir": str(ribasim_dll_dependency),
                        "config_file": str(
                            directory
                            / self._ribasim_model_dir
                            / f"{self.ribasim_model.modelname}.toml"
                        ),
                    },
                },
                "coupling": [coupling_dict],
            },
        }

        with open(toml_path, "wb") as f:
            tomli_w.dump(coupler_toml, f)

    @staticmethod
    def validate_keys(gwf_model, active_keys, passive_keys, expected_type):
        active_keys = set(active_keys)
        passive_keys = set(passive_keys)
        intersection = active_keys.intersection(passive_keys)
        if intersection:
            raise ValueError(
                f"active and passive keys contain common keys: {intersection}"
            )
        present = [k for k, v in gwf_model.items() if isinstance(v, expected_type)]
        missing = (active_keys | passive_keys).difference(present)
        if missing:
            raise ValueError(
                f"keys with expected type {expected_type.__name__} are not "
                f"present in the model: {missing}"
            )
        return

    @staticmethod
    def derive_river_drainage_coupling(
        gridded_basin: xr.DataArray, package: Union[River, Drainage]
    ) -> pd.DataFrame:
        # Conductance is leading parameter to define location, for both river
        # and drainage.
        # FUTURE: check for time dimension?
        conductance = package.dataset["conductance"]
        basin_id = gridded_basin.where(conductance.notnull(), drop=True)
        include = basin_id.notnull()
        basin_id = basin_id.to_numpy[include]
        boundary_id = np.arange(basin_id.size)[include]
        return pd.DataFrame(data={"basin_id": basin_id, "bound_id": boundary_id})

    def write_exchanges(
        self,
        directory: Union[str, Path],
    ) -> Dict:
        gwf_names = self._get_gwf_modelnames()
        coupling = self.coupling_list

        # Assume only one groundwater flow model
        # FUTURE: Support multiple groundwater flow models.
        gwf_model = self.mf6_simulation[gwf_names[0]]
        self.validate_keys(
            gwf_model,
            coupling.mf6_active_river_packages,
            coupling.mf6_passive_river_packages,
            River,
        )
        self.validate_keys(
            gwf_model,
            coupling.mf6_active_drainage_packages,
            coupling.mf6_passive_drainage_packages,
            River,
        )

        diskey = gwf_model.__get_diskey()
        dis = gwf_model[diskey]
        gridded_basin = imod.prepare.rasterize(
            self.basin_definition, like=dis["idomain"], column="basin_id"
        )

        exchange_dir = directory / "exchanges"
        exchange_dir.mkdir(mode=755, exist_ok=True)

        packages = asdict(coupling)
        packages.pop("mf6_model")
        coupling_dict = defaultdict(dict)
        for destination, keys in packages.items():
            for key in keys:
                package = gwf_model[key]
                table = self.derive_river_drainage_coupling(gridded_basin, package)
                path = exchange_dir / f"{key}.tsv"
                table.to_csv(path, sep="\t")
                coupling_dict[destination][key] = path

        return coupling_dict
