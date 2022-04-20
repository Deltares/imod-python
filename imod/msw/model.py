import collections
from copy import copy
from pathlib import Path
from typing import Union

import jinja2
import numpy as np

from imod.msw.coupler_mapping import CouplerMapping
from imod.msw.grid_data import GridData
from imod.msw.idf_mapping import IdfMapping
from imod.msw.infiltration import Infiltration
from imod.msw.initial_conditions import (
    InitialConditionsEquilibrium,
    InitialConditionsPercolation,
    InitialConditionsRootzonePressureHead,
    InitialConditionsSavedState,
)
from imod.msw.landuse import LanduseOptions
from imod.msw.meteo_grid import MeteoGrid
from imod.msw.meteo_mapping import EvapotranspirationMapping, PrecipitationMapping
from imod.msw.output_control import TimeOutputControl
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.timeutil import to_metaswap_timeformat
from imod.msw.vegetation import AnnualCropFactors

REQUIRED_PACKAGES = [
    GridData,
    CouplerMapping,
    Infiltration,
    LanduseOptions,
    MeteoGrid,
    EvapotranspirationMapping,
    PrecipitationMapping,
    IdfMapping,
    TimeOutputControl,
    AnnualCropFactors,
]

INITIAL_CONDITIONS_PACKAGES = [
    InitialConditionsEquilibrium,
    InitialConditionsPercolation,
    InitialConditionsRootzonePressureHead,
    InitialConditionsSavedState,
]

DEFAULT_SETTINGS = dict(
    vegetation_mdl=1,
    evapotranspiration_mdl=1,
    saltstress_mdl=0,
    surfacewater_mdl=0,
    infilimsat_opt=0,
    netcdf_per=0,
    postmsw_opt=0,
    dtgw=1.0,
    dtsw=1.0,
    ipstep=2,
    nxlvage_dim=366,
    co2=404.32,
    fact_beta2=1.0,
    rcsoil=0.15,
    iterur1=3,
    iterur2=5,
    tdbgsm=91.0,
    tdedsm=270.0,
    clocktime=0,
)


class Model(collections.UserDict):
    def __setitem__(self, key, value):
        # TODO: Add packagecheck
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class MetaSwapModel(Model):
    """
    Contains data and writes consistent model input files

    Parameters
    ----------
    unsaturated_database: Path-like or str
        Path to the MetaSWAP soil physical database folder.
    """

    _pkg_id = "model"
    _file_name = "para_sim.inp"

    _template = jinja2.Template(
        "{%for setting, value in settings.items()%}"
        "{{setting}} = {{value}}\n"
        "{%endfor%}"
    )

    def __init__(self, unsaturated_database):
        super().__init__()

        self.simulation_settings = copy(DEFAULT_SETTINGS)
        self.simulation_settings[
            "unsa_svat_path"
        ] = self._render_unsaturated_database_path(unsaturated_database)

    def _render_unsaturated_database_path(self, unsaturated_database):
        # Force to Path object
        unsaturated_database = Path(unsaturated_database)

        # Render to string for MetaSWAP
        if unsaturated_database.is_absolute():
            return f'"{unsaturated_database}\\"'
        else:
            # TODO: Test if this is how MetaSWAP accepts relative paths
            return f'"${unsaturated_database}\\"'

    def _check_required_packages(self):
        pkg_types_included = {type(pkg) for pkg in self.values()}
        missing_packages = set(REQUIRED_PACKAGES) - pkg_types_included
        if len(missing_packages) > 0:
            raise ValueError(
                f"Missing the following required packages: {missing_packages}"
            )

        initial_condition_set = pkg_types_included & set(INITIAL_CONDITIONS_PACKAGES)
        if len(initial_condition_set) < 1:
            raise ValueError(
                "Missing InitialCondition package, assign one of "
                f"{INITIAL_CONDITIONS_PACKAGES}"
            )
        elif len(initial_condition_set) > 1:
            raise ValueError(
                "Multiple InitialConditions assigned, choose one of "
                f"{initial_condition_set}"
            )

    def _check_landuse_indices_in_lookup_options(self):
        grid_key = self._get_pkg_key(GridData)
        landuse_options_key = self._get_pkg_key(LanduseOptions)

        indices_in_grid = set(np.unique(self[grid_key]["landuse"]))
        indices_in_options = set(
            self[landuse_options_key].dataset.coords["landuse_index"].values
        )

        missing_indices = indices_in_grid - indices_in_options

        if len(missing_indices) > 0:
            raise ValueError(
                "Found the following landuse indices in GridData which "
                f"were not in LanduseOptions: {missing_indices}"
            )

    def _check_vegetation_indices_in_annual_crop_factors(self):
        landuse_options_key = self._get_pkg_key(LanduseOptions)
        annual_crop_factors_key = self._get_pkg_key(AnnualCropFactors)

        indices_in_options = set(
            np.unique(self[landuse_options_key]["vegetation_index"])
        )
        indices_in_crop_factors = set(
            self[annual_crop_factors_key].dataset.coords["vegetation_index"].values
        )

        missing_indices = indices_in_options - indices_in_crop_factors

        if len(missing_indices) > 0:
            raise ValueError(
                "Found the following vegetation indices in LanduseOptions "
                f"which were not in AnnualCropGrowth: {missing_indices}"
            )

    def _get_starttime(self):
        """
        Loop over all packages to get the minimum time.
        """

        starttimes = []

        for pkgname in self:
            ds = self[pkgname].dataset
            if "time" in ds.coords:
                starttimes.append(ds["time"].min().values)

        starttime = min(starttimes)

        year, time_since_start_year = to_metaswap_timeformat([starttime])

        year = int(year.values)
        time_since_start_year = float(time_since_start_year.values)

        return year, time_since_start_year

    def _get_pkg_key(self, pkg_type: MetaSwapPackage, optional_package: bool = False):
        for pkg_key, pkg in self.items():
            if isinstance(pkg, pkg_type):
                return pkg_key

        if not optional_package:
            raise KeyError(f"Could not find package of type: {pkg_type}")

    def write(self, directory: Union[str, Path]):
        """
        Write packages and simulation settings (para_sim.inp).

        Parameters
        ----------
        directory: Path or str
            directory to write model in.
        """

        # Model checks
        self._check_required_packages()
        self._check_vegetation_indices_in_annual_crop_factors()
        self._check_landuse_indices_in_lookup_options()

        # Force to Path
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        # Add time settings
        year, time_since_start_year = self._get_starttime()

        self.simulation_settings["iybg"] = year
        self.simulation_settings["tdbg"] = time_since_start_year

        # Add IdfMapping settings
        idf_key = self._get_pkg_key(IdfMapping)
        self.simulation_settings.update(self[idf_key].get_output_settings())

        filename = directory / self._file_name
        with open(filename, "w") as f:
            rendered = self._template.render(settings=self.simulation_settings)
            f.write(rendered)

        # Get index and svat
        grid_key = self._get_pkg_key(GridData)
        index, svat = self[grid_key].generate_index_array()

        # write package contents
        for pkgname in self:
            self[pkgname].write(directory, index, svat)
