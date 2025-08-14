import collections
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast
import warnings

import cftime
import jinja2
import numpy as np
import xarray as xr

from imod.common.utilities.value_filters import enforce_scalar
from imod.common.utilities.version import prepend_content_with_version_info
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.msw.copy_files import FileCopier
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
from imod.msw.meteo_grid import MeteoGrid, MeteoGridCopy
from imod.msw.meteo_mapping import (
    EvapotranspirationMapping,
    MeteoMapping,
    PrecipitationMapping,
)
from imod.msw.output_control import TimeOutputControl
from imod.msw.pkgbase import MetaSwapPackage
from imod.msw.ponding import Ponding
from imod.msw.scaling_factors import ScalingFactors
from imod.msw.sprinkling import Sprinkling
from imod.msw.timeutil import to_metaswap_timeformat
from imod.msw.utilities.common import find_in_file_list
from imod.msw.utilities.imod5_converter import has_active_scaling_factor
from imod.msw.utilities.mask import MaskValues, mask_and_broadcast_cap_data
from imod.msw.utilities.parse import read_para_sim
from imod.msw.vegetation import AnnualCropFactors
from imod.typing import Imod5DataDict
from imod.util.dims import drop_layer_dim_cap_data
from imod.util.regrid import RegridderWeightsCache

REQUIRED_PACKAGES = (
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
)

INITIAL_CONDITIONS_PACKAGES = (
    InitialConditionsEquilibrium,
    InitialConditionsPercolation,
    InitialConditionsRootzonePressureHead,
    InitialConditionsSavedState,
)

DEFAULT_SETTINGS: dict[str, Any] = {
    "vegetation_mdl": 1,
    "evapotranspiration_mdl": 1,
    "saltstress_mdl": 0,
    "surfacewater_mdl": 0,
    "infilimsat_opt": 0,
    "netcdf_per": 0,
    "postmsw_opt": 0,
    "dtgw": 1.0,
    "dtsw": 1.0,
    "ipstep": 2,
    "nxlvage_dim": 366,
    "co2": 404.32,
    "fact_beta2": 1.0,
    "rcsoil": 0.15,
    "iterur1": 3,
    "iterur2": 5,
    "tdbgsm": 91.0,
    "tdedsm": 270.0,
    "clocktime": 0,
}


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
    settings: dict
    """

    _pkg_id = "model"
    _file_name = "para_sim.inp"

    _template = jinja2.Template(
        "{%for setting, value in settings.items()%}{{setting}} = {{value}}\n{%endfor%}"
    )

    def __init__(
        self,
        unsaturated_database: Path | str,
        settings: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        if settings is None:
            self.simulation_settings = copy(DEFAULT_SETTINGS)
        else:
            self.simulation_settings = settings

        self.simulation_settings["unsa_svat_path"] = unsaturated_database

    def _render_unsaturated_database_path(self, unsaturated_database: Union[str, Path]):
        # Force to Path object
        unsaturated_database = Path(unsaturated_database)

        # Render to string for MetaSWAP
        if unsaturated_database.is_absolute():
            return f'"{unsaturated_database}\\"'
        else:
            # TODO: Test if this is how MetaSWAP accepts relative paths
            return f'"${unsaturated_database}\\"'

    def _check_required_packages(self) -> None:
        pkg_types_included = {type(pkg) for pkg in self.values()}
        required_packages_set = cast(set[type[Any]], set(REQUIRED_PACKAGES))
        missing_packages = required_packages_set - pkg_types_included
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
        grid_key = self.get_pkgkey(GridData)
        landuse_options_key = self.get_pkgkey(LanduseOptions)

        indices_in_grid = set(self[grid_key]["landuse"].values.ravel())
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
        landuse_options_key = self.get_pkgkey(LanduseOptions)
        annual_crop_factors_key = self.get_pkgkey(AnnualCropFactors)

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

    def _has_file_copier(self) -> bool:
        pkg_types_included = {type(pkg) for pkg in self.values()}
        return FileCopier in pkg_types_included

    def _get_starttime(self):
        """
        Loop over all packages to get the minimum time.

        MetaSWAP requires a starttime in its simulation settings (para_sim.inp)
        """

        starttimes = []

        for pkgname in self:
            ds = self[pkgname].dataset
            if "time" in ds.coords:
                starttimes.append(ds["time"].min().values)

        starttime = min(starttimes)

        year, time_since_start_year = to_metaswap_timeformat([starttime])

        year = int(enforce_scalar(year.values))
        time_since_start_year = float(enforce_scalar(time_since_start_year.values))

        return year, time_since_start_year

    def get_pkgkey(self, pkg_type: type[MetaSwapPackage], optional_package: bool = False) -> str:
        """
        Get the package key for a package of type ``pkg_type``. Returns the
        first occurrence of the package type.

        Parameters
        ----------
        pkg_type: type[MetaSwapPackage]
            Type of the package to get the key for.
        optional_package: bool
            If True, the method will not raise an error if the package is not found.
        
        Returns
        -------
        str
            The key of the package of type ``pkg_type``.
        """
        for pkg_key, pkg in self.items():
            if isinstance(pkg, pkg_type):
                return pkg_key

        if not optional_package:
            raise KeyError(f"Could not find package of type: {pkg_type}")

    def _get_pkg_key(self, pkg_type: type[MetaSwapPackage], optional_package: bool = False):
        """"
        Preserves backwards compatibility with old code (primod) that used this.
        """
        warnings.warn(
            "Method '_get_pkg_key' is deprecated, use 'get_pkgkey' instead.",
            DeprecationWarning,
        )
        return self.get_pkgkey(pkg_type, optional_package)

    def _model_checks(self, validate: bool):
        if validate and not self._has_file_copier():
            self._check_required_packages()
            self._check_vegetation_indices_in_annual_crop_factors()
            self._check_landuse_indices_in_lookup_options()

    def _write_simulation_settings(self, directory: Path) -> None:
        """
        Write simulation settings to para_sim.inp.

        Parameters
        ----------
        directory: Path or str
            directory to write model in.
        """
        simulation_settings = deepcopy(self.simulation_settings)

        # Add time settings
        year, time_since_start_year = self._get_starttime()

        simulation_settings["iybg"] = year
        simulation_settings["tdbg"] = time_since_start_year

        # Add IdfMapping settings
        idf_key = self.get_pkgkey(IdfMapping)
        idf_pkg = cast(IdfMapping, self[idf_key])
        simulation_settings.update(idf_pkg._get_output_settings())

        simulation_settings["unsa_svat_path"] = self._render_unsaturated_database_path(
            simulation_settings["unsa_svat_path"]
        )

        filename = directory / self._file_name
        rendered = self._template.render(settings=simulation_settings)
        # Prepend version information
        rendered = prepend_content_with_version_info(rendered, comment_char="*")
        with open(filename, "w") as f:
            f.write(rendered)

    def write(
        self,
        directory: Union[str, Path],
        mf6_dis: StructuredDiscretization,
        mf6_wel: Mf6Wel,
        validate: bool = True,
    ):
        """
        Write packages and simulation settings (para_sim.inp).

        Parameters
        ----------
        directory: Path or str
            directory to write model in.
        """

        # Model checks
        self._model_checks(validate)

        # Force to Path
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        # Write simulation settings
        self._write_simulation_settings(directory)

        # Get index and svat
        grid_key = self.get_pkgkey(GridData)
        grid_pkg = cast(GridData, self[grid_key])
        index, svat = grid_pkg.generate_index_array()

        # write package contents
        for pkgname in self:
            self[pkgname].write(directory, index, svat, mf6_dis, mf6_wel)

    def regrid_like(
        self,
        mf6_regridded_dis: StructuredDiscretization,
        regrid_cache: Optional[RegridderWeightsCache] = None,
    ) -> "MetaSwapModel":
        """
        Creates a model by regridding the packages of this model to another
        discretization. It regrids all the arrays in the package using the
        default regridding methods. At the moment only regridding to a different
        planar grid is supported, meaning ``target_grid`` has different ``"x"``
        and ``"y"`` or different ``cell2d`` coords.

        Parameters
        ----------
        mf6_regridded_dis: StructuredDiscretization
            Modflow6 Discretization with same discretization as the one we want
            to regrid the package to.
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to
            speed up regridding, if the same regridders are used several times
            for regridding different arrays.

        Returns
        -------
        A model with similar packages to the input model, and with all the
        data-arrays regridded to another discretization, similar to the one used
        in input argument "mf6_regridded_dis"
        """
        unsat_database = cast(str, self.simulation_settings["unsa_svat_path"])
        regridded_model = MetaSwapModel(unsat_database)

        target_grid = mf6_regridded_dis["idomain"]

        mod2svat_name = None

        for pkgname in self:
            msw_package = self[pkgname]
            if isinstance(msw_package, CouplerMapping):
                # there can be only one couplermapping
                mod2svat_name = pkgname
            elif msw_package._is_regridding_supported():
                regridded_package = msw_package.regrid_like(
                    target_grid, regrid_cache, None
                )
            else:
                raise ValueError(f"package {pkgname} cannot be  regridded")
            regridded_model[pkgname] = regridded_package
        if mod2svat_name is not None:
            regridded_model[mod2svat_name] = CouplerMapping()

        return regridded_model

    def clip_box(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> "MetaSwapModel":
        """
        Clip a model by a bounding box (time, y, x). If a package of type
        :class:`imod.msw.MeteoGridCopy` is present, packages of type
        :class:`imod.msw.PrecipitationMapping` and
        :class:`imod.msw.EvapotranspirationMapping` will not be clipped.
        Otherwise incorrect mappings to meteo grids referenced in
        ``mete_grid.inp`` copied by :class:`imod.msw.MeteoGridCopy` would be
        computed.

        Parameters
        ----------
        time_min: optional, np.datetime64
            Start time to select. Data will be forward filled to this date. If
            time_min is before the start time of the dataset, data is
            backfilled.
        time_max: optional
            End time to select.
        x_min: optional, float
            Minimum x-coordinate to select.
        x_max: optional, float
            Maximum x-coordinate to select.
        y_min: optional, float
            Minimum y-coordinate to select.
        y_max: optional, float
            Maximum y-coordinate to select.

        Returns
        -------
        clipped : MetaSwapModel
            A new model that is clipped to the specified bounding box.

        Examples
        --------
        Slicing intervals may be half-bounded, by providing None:

        To select 500.0 <= x <= 1000.0:

        >>> msw_model.clip_box(x_min=500.0, x_max=1000.0)

        To select x <= 1000.0:

        >>> msw_model.clip_box(x_max=1000.0)``

        To select x >= 500.0:

        >>> msw_model.clip_box(x_min=500.0)

        To select a time interval, you can use datetime64:

        >>> msw_model.clip_box(time_min=np.datetime64("2020-01-01"), time_max=np.datetime64("2020-12-31"))

        """
        settings = deepcopy(self.simulation_settings)
        unsa_svat_path = settings.pop("unsa_svat_path")

        has_meteogrid_copy = MeteoGridCopy in [type(pkg) for pkg in self.values()]
        clipped = type(self)(unsa_svat_path, settings)
        for key, pkg in self.items():
            # Skip clipping MeteoMapping if MeteoGridCopy is present, as meteo
            # grid is independent of model grid and we do not want to perform
            # transformations on meteo data in this case.
            if has_meteogrid_copy and isinstance(pkg, MeteoMapping):
                clipped[key] = deepcopy(pkg)
            else:
                clipped[key] = pkg.clip_box(
                    time_min=time_min,
                    time_max=time_max,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
        return clipped

    @classmethod
    def from_imod5_data(
        cls,
        imod5_data: Imod5DataDict,
        target_dis: StructuredDiscretization,
        times: list[datetime],
    ):
        """
        Construct a MetaSWAP model from iMOD5 data in the CAP package, loaded
        with the :func:`imod.formats.prj.open_projectfile_data` function.

        Parameters
        ----------
        imod5_data: dict
            Dictionary with iMOD5 data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        target_dis: imod.mf6.StructuredDiscretization
            Target discretization, cells where MODLOW6 is inactive will be
            inactive in MetaSWAP as well.
        times: list[datetime]
            List of datetimes, will be used to set the output control times.
            Is also used to infer the starttime of the simulation.

        Returns
        -------
        MetaSwapModel
            MetaSWAP model imported from imod5 data.
        """
        extra_paths = imod5_data["extra"]["paths"]
        path_to_parasim = find_in_file_list("para_sim.inp", extra_paths)
        parasim_settings = read_para_sim(path_to_parasim)
        unsa_svat_path = cast(str, parasim_settings["unsa_svat_path"])
        # Drop layer coord
        imod5_cap_no_layer = drop_layer_dim_cap_data(imod5_data)
        model = cls(unsa_svat_path, parasim_settings)
        model["grid"], msw_active = GridData.from_imod5_data(
            imod5_cap_no_layer, target_dis
        )
        cap_data_masked = mask_and_broadcast_cap_data(
            imod5_cap_no_layer["cap"], msw_active
        )
        imod5_masked: Imod5DataDict = {
            "cap": cap_data_masked,
            "extra": {"paths": extra_paths},
        }
        model["infiltration"] = Infiltration.from_imod5_data(imod5_masked)
        model["ponding"] = Ponding.from_imod5_data(imod5_masked)
        model["sprinkling"] = Sprinkling.from_imod5_data(imod5_masked)
        model["meteo_grid"] = MeteoGridCopy.from_imod5_data(imod5_masked)
        model["prec_mapping"] = PrecipitationMapping.from_imod5_data(imod5_masked)
        model["evt_mapping"] = EvapotranspirationMapping.from_imod5_data(imod5_masked)
        if has_active_scaling_factor(imod5_cap_no_layer["cap"]):
            model["scaling_factor"] = ScalingFactors.from_imod5_data(imod5_masked)
        area = model["grid"]["area"].isel(subunit=0, drop=True)
        model["idf_mapping"] = IdfMapping(area, MaskValues.default)
        model["coupling"] = CouplerMapping()
        model["extra_files"] = FileCopier.from_imod5_data(imod5_masked)

        times_da = xr.DataArray(times, coords={"time": times}, dims=("time",))
        model["time_oc"] = TimeOutputControl(times_da)

        return model
