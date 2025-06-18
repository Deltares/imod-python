from __future__ import annotations

import abc
import collections
import inspect
import pathlib
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cftime
import jinja2
import numpy as np
import tomli
import tomli_w
import xarray as xr
import xugrid as xu
from jinja2 import Template

import imod
from imod.common.interfaces.imodel import IModel
from imod.common.statusinfo import NestedStatusInfo, StatusInfo, StatusInfoBase
from imod.common.utilities.mask import _mask_all_packages
from imod.common.utilities.regrid import _regrid_like
from imod.common.utilities.schemata import (
    concatenate_schemata_dicts,
    pkg_errors_to_status_info,
    validate_schemata_dict,
    validate_with_error_message,
)
from imod.common.utilities.version import prepend_content_with_version_info
from imod.logging import LogLevel, logger, standard_log_decorator
from imod.mf6.drn import Drainage
from imod.mf6.ghb import GeneralHeadBoundary
from imod.mf6.hfb import HorizontalFlowBarrierBase
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.mf6.package import Package
from imod.mf6.riv import River
from imod.mf6.utilities.mf6hfb import merge_hfb_packages
from imod.mf6.validation_context import ValidationContext
from imod.mf6.wel import GridAgnosticWell
from imod.mf6.write_context import WriteContext
from imod.schemata import SchemataDict, ValidationError
from imod.typing import GridDataArray
from imod.util.regrid import RegridderWeightsCache

HFB_PKGNAME = "hfb_merged"
SUGGESTION_TEXT = (
    "-> You might fix this by calling the package's ``.cleanup()`` method."
)
PKGTYPES_WITH_CLEANUP = [River, Drainage, GeneralHeadBoundary, GridAgnosticWell]


def pkg_has_cleanup(pkg: Package):
    return any(isinstance(pkg, pkgtype) for pkgtype in PKGTYPES_WITH_CLEANUP)


class Modflow6Model(collections.UserDict, IModel, abc.ABC):
    _mandatory_packages: tuple[str, ...] = ()
    _init_schemata: SchemataDict = {}
    _model_id: Optional[str] = None
    _template: Template

    @staticmethod
    def _initialize_template(name: str) -> Template:
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        return env.get_template(name)

    def __init__(self):
        collections.UserDict.__init__(self)
        self._options = {}

    @standard_log_decorator()
    def validate_options(
        self, schemata: dict, **kwargs
    ) -> dict[str, list[ValidationError]]:
        return validate_schemata_dict(schemata, self._options, **kwargs)

    def validate_init_schemata_options(self, validate: bool) -> None:
        """
        Run the "cheap" schema validations.

        The expensive validations are run during writing. Some are only
        available then: e.g. idomain to determine active part of domain.
        """
        validate_with_error_message(
            self.validate_options, validate, self._init_schemata
        )

    def __setitem__(self, key, value):
        if len(key) > 16:
            raise KeyError(
                f"Received key with more than 16 characters: '{key}'"
                "Modflow 6 has a character limit of 16."
            )

        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def _get_diskey(self):
        dis_pkg_ids = ["dis", "disv", "disu"]

        diskeys = [
            self._get_pkgkey(pkg_id)
            for pkg_id in dis_pkg_ids
            if self._get_pkgkey(pkg_id) is not None
        ]

        if len(diskeys) > 1:
            raise ValueError(f"Found multiple discretizations {diskeys}")
        elif len(diskeys) == 0:
            raise ValueError("No model discretization found")
        else:
            return diskeys[0]

    def _get_pkgkey(self, pkg_id):
        """
        Get package key that belongs to a certain pkg_id, since the keys are
        user specified.
        """
        key = [pkgname for pkgname, pkg in self.items() if pkg._pkg_id == pkg_id]
        nkey = len(key)
        if nkey > 1:
            raise ValueError(f"Multiple instances of {key} detected")
        elif nkey == 1:
            return key[0]
        else:
            return None

    def _check_for_required_packages(self, modelkey: str) -> None:
        # Check for mandatory packages
        pkg_ids = {pkg._pkg_id for pkg in self.values()}
        dispresent = "dis" in pkg_ids or "disv" in pkg_ids or "disu" in pkg_ids
        if not dispresent:
            raise ValueError(f"No dis/disv/disu package found in model {modelkey}")
        for required in self._mandatory_packages:
            if required not in pkg_ids:
                raise ValueError(f"No {required} package found in model {modelkey}")
        return

    def _use_cftime(self):
        """
        Also checks if datetime types are homogeneous across packages.
        """
        types = [
            type(pkg.dataset["time"].values[0])
            for pkg in self.values()
            if "time" in pkg.dataset.coords
        ]
        set_of_types = set(types)
        # Types will be empty if there's no time dependent input
        if len(set_of_types) == 0:
            return False
        else:  # there is time dependent input
            if not len(set_of_types) == 1:
                raise ValueError(
                    f"Multiple datetime types detected: {set_of_types}"
                    "Use either cftime or numpy.datetime64[ns]."
                )
            # Since we compare types and not instances, we use issubclass
            if issubclass(types[0], cftime.datetime):
                return True
            elif issubclass(types[0], np.datetime64):
                return False
            else:
                raise ValueError("Use either cftime or numpy.datetime64[ns].")

    def _yield_times(self):
        modeltimes = []
        for pkg in self.values():
            if "time" in pkg.dataset.coords:
                modeltimes.append(pkg.dataset["time"].values)
            repeat_stress = pkg.dataset.get("repeat_stress")
            if repeat_stress is not None and repeat_stress.values[()] is not None:
                modeltimes.append(repeat_stress.isel(repeat_items=0).values)
        return modeltimes

    def render(self, modelname: str, write_context: WriteContext):
        dir_for_render = write_context.root_directory / modelname

        d = {k: v for k, v in self._options.items() if not (v is None or v is False)}
        packages = []
        has_hfb = False
        for pkgname, pkg in self.items():
            # Add the six to the package id
            pkg_id = pkg._pkg_id
            # Skip if hfb
            if pkg_id == "hfb":
                has_hfb = True
                continue
            key = f"{pkg_id}6"
            path = dir_for_render / f"{pkgname}.{pkg_id}"
            packages.append((key, path.as_posix(), pkgname))
        if has_hfb:
            path = dir_for_render / f"{HFB_PKGNAME}.hfb"
            packages.append(("hfb6", path.as_posix(), HFB_PKGNAME))
        d["packages"] = packages
        return self._template.render(d)

    def _model_checks(self, modelkey: str):
        """
        Check model integrity (called before writing)
        """

        self._check_for_required_packages(modelkey)

    def _get_domain_geometry(
        self,
    ) -> tuple[
        Union[xr.DataArray, xu.UgridDataArray],
        Union[xr.DataArray, xu.UgridDataArray],
        Union[xr.DataArray, xu.UgridDataArray],
    ]:
        discretization = self[self._get_diskey()]
        if discretization is None:
            raise ValueError("Discretization not found")
        top = discretization["top"]
        bottom = discretization["bottom"]
        idomain = discretization["idomain"]
        return top, bottom, idomain

    def _get_k(self):
        npf_key = self._get_pkgkey(imod.mf6.NodePropertyFlow._pkg_id)
        if not npf_key:
            raise KeyError(
                """Unable to obtain the hydraulic conductivity. Make sure a
                NodePropertyFlow package is assigned to the Modflow6Model."""
            )
        npf = self[npf_key]
        return npf["k"]

    @standard_log_decorator()
    def validate(
        self,
        model_name: str = "",
        validation_context: Optional[ValidationContext] = None,
    ) -> StatusInfoBase:
        if validation_context is None:
            validation_context = ValidationContext(validate=True)

        try:
            diskey = self._get_diskey()
        except Exception as e:
            status_info = StatusInfo(f"{model_name} model")
            status_info.add_error(str(e))
            return status_info

        dis = self[diskey]
        # We'll use the idomain for checking dims, shape, nodata.
        idomain = dis["idomain"]
        bottom = dis["bottom"]

        model_status_info = NestedStatusInfo(f"{model_name} model")
        # Check model options
        option_errors = self.validate_options(self._init_schemata)
        model_status_info.add(
            pkg_errors_to_status_info("model options", option_errors, None)
        )
        # Validate packages
        for pkg_name, pkg in self.items():
            # Check for all schemata when writing. Types and dimensions
            # may have been changed after initialization...

            if pkg_name in ["adv"]:
                continue  # some packages can be skipped

            # Concatenate write and init schemata.
            schemata = concatenate_schemata_dicts(
                pkg._init_schemata, pkg._write_schemata
            )

            pkg_errors = pkg._validate(
                schemata=schemata,
                idomain=idomain,
                bottom=bottom,
                validation_context=validation_context,
            )
            if len(pkg_errors) > 0:
                footer = SUGGESTION_TEXT if pkg_has_cleanup(pkg) else None
                model_status_info.add(
                    pkg_errors_to_status_info(pkg_name, pkg_errors, footer)
                )

        return model_status_info

    def prepare_wel_for_mf6(
        self, pkgname: str, validate: bool = True, strict_well_validation: bool = True
    ) -> Mf6Wel:
        """
        Prepare grid-agnostic well for MODFLOW6, using the models grid
        information and hydraulic conductivities. Allocates LayeredWell & Well
        objects, which have x,y locations, to row & columns. Furthermore, Well
        objects are allocated to model layers, depending on screen depth.

        This function is called upon writing the model, it is included in the
        public API for the user's debugging purposes.

        Parameters
        ----------
        pkgname: string
            Name of well package that is to be prepared for MODFLOW6
        validate: bool, default True
            Run validation before converting
        strict_well_validation: bool, default True
            Set well validation strict:
            Throw error if well is removed entirely during its assignment to
            layers.

        Returns
        -------
        Mf6Wel
            Direct representation of MODFLOW6 WEL package, with 'cellid'
            indicating layer, row columns.
        """
        validate_context = ValidationContext(
            validate=validate, strict_well_validation=strict_well_validation
        )
        return self._prepare_wel_for_mf6(pkgname, validate_context)

    @standard_log_decorator()
    def _prepare_wel_for_mf6(
        self, pkgname: str, validate_context: ValidationContext
    ) -> Mf6Wel:
        pkg = self[pkgname]
        if not isinstance(pkg, GridAgnosticWell):
            raise TypeError(
                f"""Package '{pkgname}' not of appropriate type, should be of type:
                 'Well', 'LayeredWell', got {type(pkg)}"""
            )
        top, bottom, idomain = self._get_domain_geometry()
        k = self._get_k()
        return pkg._to_mf6_pkg(
            idomain,
            top,
            bottom,
            k,
            validate_context,
        )

    @standard_log_decorator()
    def write(
        self,
        modelname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        directory: str | Path,
        use_binary: bool = True,
        use_absolute_paths: bool = False,
        validate: bool = True,
    ):
        """
        Write MODFLOW6 model to file. Note that this method is purely intended
        for debugging purposes. It does not result in a functional model. For
        that the model needs to be part of a
        :class:`imod.mf6.Modflow6Simulation`, of which the ``write`` method
        should be called.

        Parameters
        ----------
        modelname: str
            Model name
        globaltimes: list[np.datetime64] | np.ndarray
            Times of the simulation's stress periods.
        directory: str | Path
            Directory in which the simulation will be written.
        use_binary: bool = True
            Whether to write time-dependent input for stress packages as binary
            files, which are smaller in size, or more human-readable text files.
        use_absolute_paths: bool = False
            True if all paths written to the mf6 inputfiles should be absolute.
        validate: bool = True
            Whether to validate the Modflow6 simulation, including models, at
            write. If True, erronous model input will throw a
            ``ValidationError``.
        """
        write_context = WriteContext(Path(directory), use_binary, use_absolute_paths)
        validate_context = ValidationContext(validate, validate, validate)

        status_info = self._write(
            modelname, globaltimes, write_context, validate_context
        )

        if status_info.has_errors():
            raise ValidationError("\n" + status_info.to_string())

    @standard_log_decorator()
    def _write(
        self,
        modelname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        write_context: WriteContext,
        validate_context: ValidationContext,
    ) -> StatusInfoBase:
        """
        Write model namefile
        Write packages
        """

        workdir = write_context.simulation_directory
        modeldirectory = workdir / modelname
        Path(modeldirectory).mkdir(exist_ok=True, parents=True)
        if validate_context.validate:
            model_status_info = self.validate(modelname, validate_context)
            if model_status_info.has_errors():
                return model_status_info

        # write model namefile
        namefile_content = self.render(modelname, write_context)
        namefile_content = prepend_content_with_version_info(namefile_content)
        namefile_path = modeldirectory / f"{modelname}.nam"
        with open(namefile_path, "w") as f:
            f.write(namefile_content)

        # write package contents
        pkg_write_context = write_context.copy_with_new_write_directory(
            new_write_directory=modeldirectory
        )
        mf6_hfb_ls: List[HorizontalFlowBarrierBase] = []
        for pkg_name, pkg in self.items():
            try:
                if issubclass(type(pkg), GridAgnosticWell):
                    mf6_well_pkg = self._prepare_wel_for_mf6(pkg_name, validate_context)
                    mf6_well_pkg._write(
                        pkgname=pkg_name,
                        globaltimes=globaltimes,
                        write_context=pkg_write_context,
                    )
                elif issubclass(type(pkg), imod.mf6.HorizontalFlowBarrierBase):
                    mf6_hfb_ls.append(pkg)
                else:
                    pkg._write(
                        pkgname=pkg_name,
                        globaltimes=globaltimes,
                        write_context=pkg_write_context,
                    )
            except Exception as e:
                raise type(e)(f"{e}\nError occured while writing {pkg_name}")

        if len(mf6_hfb_ls) > 0:
            try:
                pkg_name = HFB_PKGNAME
                top, bottom, idomain = self._get_domain_geometry()
                k = self._get_k()
                mf6_hfb_pkg = merge_hfb_packages(
                    mf6_hfb_ls,
                    idomain,
                    top,
                    bottom,
                    k,
                    validate_context.strict_hfb_validation,
                )
                if len(mf6_hfb_pkg["cell_id"]) > 0:
                    mf6_hfb_pkg._write(
                        pkgname=pkg_name,
                        globaltimes=globaltimes,
                        write_context=pkg_write_context,
                    )
                else:
                    message = "No HorizontalFlowBarriers could be snapped to grids."
                    logger.log(LogLevel.WARNING, message)
            except Exception as e:
                raise type(e)(f"{e}\nError occured while writing {pkg_name}")

        return NestedStatusInfo(modelname)

    @standard_log_decorator()
    def dump(
        self,
        directory,
        modelname,
        validate: bool = True,
        mdal_compliant: bool = False,
        crs: Optional[Any] = None,
    ):
        """
        Dump simulation to files. Writes a model definition as .TOML file, which
        points to data for each package. Each package is stored as a separate
        NetCDF. Structured grids are saved as regular NetCDFs, unstructured
        grids are saved as UGRID NetCDF. Structured grids are always made GDAL
        compliant, unstructured grids can be made MDAL compliant optionally.

        Parameters
        ----------
        directory: str or Path
            directory to dump simulation into.
        modelname: str
            modelname, will be used to create a subdirectory.
        validate: bool, optional
            Whether to validate simulation data. Defaults to True.
        mdal_compliant: bool, optional
            Convert data with
            :func:`imod.prepare.spatial.mdal_compliant_ugrid2d` to MDAL
            compliant unstructured grids. Defaults to False.
        crs: Any, optional
            Anything accepted by rasterio.crs.CRS.from_user_input
            Requires ``rioxarray`` installed.
        """
        modeldirectory = pathlib.Path(directory) / modelname
        modeldirectory.mkdir(exist_ok=True, parents=True)
        validation_context = ValidationContext(validate=validate)
        if validation_context.validate:
            statusinfo = self.validate(modelname, validation_context)
            if statusinfo.has_errors():
                raise ValidationError(statusinfo.to_string())

        toml_content: dict = collections.defaultdict(dict)
        for pkgname, pkg in self.items():
            pkg_path = f"{pkgname}.nc"
            toml_content[type(pkg).__name__][pkgname] = pkg_path
            pkg.to_netcdf(
                modeldirectory / pkg_path, crs=crs, mdal_compliant=mdal_compliant
            )

        toml_path = modeldirectory / f"{modelname}.toml"
        with open(toml_path, "wb") as f:
            tomli_w.dump(toml_content, f)

        return toml_path

    @classmethod
    def from_file(cls, toml_path):
        pkg_classes = {
            name: pkg_cls
            for name, pkg_cls in inspect.getmembers(imod.mf6, inspect.isclass)
            if issubclass(pkg_cls, Package)
        }

        toml_path = pathlib.Path(toml_path)
        with open(toml_path, "rb") as f:
            toml_content = tomli.load(f)

        parentdir = toml_path.parent
        instance = cls()
        for key, entry in toml_content.items():
            for pkgname, path in entry.items():
                pkg_cls = pkg_classes[key]
                instance[pkgname] = pkg_cls.from_file(parentdir / path)

        return instance

    @property
    def options(self) -> dict:
        if self._options is None:
            raise ValueError("Model id has not been set")
        return self._options

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            raise ValueError("Model id has not been set")
        return self._model_id

    def clip_box(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        layer_min: Optional[int] = None,
        layer_max: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        state_for_boundary: Optional[GridDataArray] = None,
    ):
        """
        Clip a model by a bounding box (time, layer, y, x).

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
        state_for_boundary: optional, float
        """
        supported, error_with_object = self.is_clipping_supported()
        if not supported:
            raise ValueError(
                f"model cannot be clipped due to presence of package '{error_with_object}' in model"
            )

        clipped = self._clip_box_packages(
            time_min,
            time_max,
            layer_min,
            layer_max,
            x_min,
            x_max,
            y_min,
            y_max,
        )

        return clipped

    def _clip_box_packages(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        layer_min: Optional[int] = None,
        layer_max: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ):
        """
        Clip a model by a bounding box (time, layer, y, x).

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

        Returns
        -------
        clipped : Modflow6Model
        """

        top, bottom, _ = self._get_domain_geometry()

        clipped = type(self)(**self._options)
        for key, pkg in self.items():
            clipped[key] = pkg.clip_box(
                time_min=time_min,
                time_max=time_max,
                layer_min=layer_min,
                layer_max=layer_max,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                top=top,
                bottom=bottom,
            )

        return clipped

    def regrid_like(
        self,
        target_grid: GridDataArray,
        validate: bool = True,
        regrid_cache: Optional[RegridderWeightsCache] = None,
    ) -> "Modflow6Model":
        """
        Creates a model by regridding the packages of this model to another
        discretization. It regrids all the arrays in the package using the
        default regridding methods. At the moment only regridding to a different
        planar grid is supported, meaning ``target_grid`` has different ``"x"``
        and ``"y"`` or different ``cell2d`` coords.

        Parameters
        ----------
        target_grid: xr.DataArray or xu.UgridDataArray
            a grid defined using the same discretization as the one we want to
            regrid the package to.
        validate: bool
            set to true to validate the regridded packages
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to
            speed up regridding, if the same regridders are used several times
            for regridding different arrays.

        Returns
        -------

        a model with similar packages to the input model, and with all the
        data-arrays regridded to another discretization, similar to the one used
        in input argument "target_grid"
        """
        return _regrid_like(self, target_grid, validate, regrid_cache)

    def mask_all_packages(
        self,
        mask: GridDataArray,
    ):
        """
        This function applies a mask to all packages in a model. The mask must
        be presented as an idomain-like integer array that has 0 (inactive) or
        <0 (vertical passthrough) values in filtered cells and >0 in active
        cells.
        Masking will overwrite idomain with the mask where the mask is <=0.
        Where the mask is >0, the original value of idomain will be kept. Masking
        will update the packages accordingly, blanking their input where needed,
        and is therefore not a reversible operation.

        Parameters
        ----------
        mask: xr.DataArray, xu.UgridDataArray of ints
            idomain-like integer array. >0 sets cells to active, 0 sets cells to inactive,
            <0 sets cells to vertical passthrough
        """

        _mask_all_packages(self, mask)

    def purge_empty_packages(
        self, model_name: Optional[str] = "", ignore_time: bool = False
    ) -> None:
        """
        This function removes empty packages from the model.
        """
        empty_packages = [
            package_name
            for package_name, package in self.items()
            if package.is_empty(ignore_time=ignore_time)
        ]
        logger.info(
            f"packages: {empty_packages} removed in {model_name}, because all empty"
        )
        for package_name in empty_packages:
            self.pop(package_name)

    @property
    def domain(self):
        dis = self._get_diskey()
        return self[dis]["idomain"]

    @property
    def bottom(self):
        dis = self._get_diskey()
        return self[dis]["bottom"]

    def __repr__(self) -> str:
        INDENT = "    "
        typename = type(self).__name__
        options = [
            f"{INDENT}{key}={repr(value)}," for key, value in self._options.items()
        ]
        packages = [
            f"{INDENT}{repr(key)}: {type(value).__name__},"
            for key, value in self.items()
        ]
        # Place the emtpy dict on the same line. Looks silly otherwise.
        if packages:
            content = [f"{typename}("] + options + ["){"] + packages + ["}"]
        else:
            content = [f"{typename}("] + options + ["){}"]
        return "\n".join(content)

    def is_use_newton(self):
        return False

    def is_splitting_supported(self) -> Tuple[bool, str]:
        """
        Returns True if all the packages in the model supports splitting. If one
        of the packages in the model does not support splitting, it returns the
        name of the first one.
        """
        for package_name, package in self.items():
            if not package.is_splitting_supported():
                return False, package_name
        return True, ""

    def is_regridding_supported(self) -> Tuple[bool, str]:
        """
        Returns True if all the packages in the model supports regridding. If one
        of the packages in the model does not support regridding, it returns the
        name of the first one.
        """
        for package_name, package in self.items():
            if not package.is_regridding_supported():
                return False, package_name
        return True, ""

    def is_clipping_supported(self) -> Tuple[bool, str]:
        """
        Returns True if all the packages in the model supports clipping. If one
        of the packages in the model does not support clipping, it returns the
        name of the first one.
        """
        for package_name, package in self.items():
            if not package.is_clipping_supported():
                return False, package_name
        return True, ""
