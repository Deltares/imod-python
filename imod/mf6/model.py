from __future__ import annotations

import abc
import collections
import inspect
import pathlib
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import cftime
import jinja2
import numpy as np
import tomli
import tomli_w
import xarray as xr
import xugrid as xu
from jinja2 import Template

import imod
from imod.logging.logging_decorators import standard_log_decorator
from imod.mf6.interfaces.imodel import IModel
from imod.mf6.package import Package
from imod.mf6.statusinfo import NestedStatusInfo, StatusInfo, StatusInfoBase
from imod.mf6.utilities.mask import _mask_all_packages
from imod.mf6.utilities.regrid import (
    _regrid_like,
)
from imod.mf6.validation import pkg_errors_to_status_info
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.typing import GridDataArray


class Modflow6Model( collections.UserDict, IModel, abc.ABC):
    _mandatory_packages: tuple[str, ...] = ()
    _model_id: Optional[str] = None
    _template: Template

    @staticmethod
    def _initialize_template(name: str) -> Template:
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        return env.get_template(name)


    def __init__(self, **kwargs):
        collections.UserDict.__init__(self)
        for k, v in kwargs.items():
            self[k] = v

        self._options = {}

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
        pkg_ids = set([pkg._pkg_id for pkg in self.values()])
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
        for pkgname, pkg in self.items():
            # Add the six to the package id
            pkg_id = pkg._pkg_id
            key = f"{pkg_id}6"
            path = dir_for_render / f"{pkgname}.{pkg_id}"
            packages.append((key, path.as_posix(), pkgname))
        d["packages"] = packages
        return self._template.render(d)

    def _model_checks(self, modelkey: str):
        """
        Check model integrity (called before writing)
        """

        self._check_for_required_packages(modelkey)

    def __get_domain_geometry(
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

    def __get_k(self):
        try:
            npf = self[imod.mf6.NodePropertyFlow._pkg_id]
        except RuntimeError:
            raise ValidationError("expected one package of type ModePropertyFlow")

        k = npf["k"]
        return k
    
    @standard_log_decorator()
    def validate(self, model_name: str = "") -> StatusInfoBase:
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
        for pkg_name, pkg in self.items():
            # Check for all schemata when writing. Types and dimensions
            # may have been changed after initialization...

            if pkg_name in ["adv"]:
                continue  # some packages can be skipped

            # Concatenate write and init schemata.
            schemata = deepcopy(pkg._init_schemata)
            for key, value in pkg._write_schemata.items():
                if key not in schemata.keys():
                    schemata[key] = value
                else:
                    schemata[key] += value

            pkg_errors = pkg._validate(
                schemata=schemata,
                idomain=idomain,
                bottom=bottom,
            )
            if len(pkg_errors) > 0:
                model_status_info.add(pkg_errors_to_status_info(pkg_name, pkg_errors))

        return model_status_info
    
    @standard_log_decorator()
    def write(
        self, modelname, globaltimes, validate: bool, write_context: WriteContext
    ) -> StatusInfoBase:
        """
        Write model namefile
        Write packages
        """

        workdir = write_context.simulation_directory
        modeldirectory = workdir / modelname
        Path(modeldirectory).mkdir(exist_ok=True, parents=True)
        if validate:
            model_status_info = self.validate(modelname)
            if model_status_info.has_errors():
                return model_status_info

        # write model namefile
        namefile_content = self.render(modelname, write_context)
        namefile_path = modeldirectory / f"{modelname}.nam"
        with open(namefile_path, "w") as f:
            f.write(namefile_content)

        # write package contents
        pkg_write_context = write_context.copy_with_new_write_directory(
            new_write_directory=modeldirectory
        )
        for pkg_name, pkg in self.items():
            try:
                if isinstance(pkg, imod.mf6.Well):
                    top, bottom, idomain = self.__get_domain_geometry()
                    k = self.__get_k()
                    mf6_well_pkg = pkg.to_mf6_pkg(
                        idomain,
                        top,
                        bottom,
                        k,
                        validate,
                        pkg_write_context.is_partitioned,
                    )

                    mf6_well_pkg.write(
                        pkgname=pkg_name,
                        globaltimes=globaltimes,
                        write_context=pkg_write_context,
                    )
                elif isinstance(pkg, imod.mf6.HorizontalFlowBarrierBase):
                    top, bottom, idomain = self.__get_domain_geometry()
                    k = self.__get_k()
                    mf6_hfb_pkg = pkg.to_mf6_pkg(idomain, top, bottom, k, validate)
                    mf6_hfb_pkg.write(
                        pkgname=pkg_name,
                        globaltimes=globaltimes,
                        write_context=pkg_write_context,
                    )
                else:
                    pkg.write(
                        pkgname=pkg_name,
                        globaltimes=globaltimes,
                        write_context=pkg_write_context,
                    )
            except Exception as e:
                raise type(e)(f"{e}\nError occured while writing {pkg_name}")

        return NestedStatusInfo(modelname)

    @standard_log_decorator()
    def dump(
        self, directory, modelname, validate: bool = True, mdal_compliant: bool = False
    ):
        modeldirectory = pathlib.Path(directory) / modelname
        modeldirectory.mkdir(exist_ok=True, parents=True)
        if validate:
            statusinfo = self.validate()
            if statusinfo.has_errors():
                raise ValidationError(statusinfo.to_string())

        toml_content: dict = collections.defaultdict(dict)
        for pkgname, pkg in self.items():
            pkg_path = f"{pkgname}.nc"
            toml_content[type(pkg).__name__][pkgname] = pkg_path
            dataset = pkg.dataset
            if isinstance(dataset, xu.UgridDataset):
                if mdal_compliant:
                    dataset = pkg.dataset.ugrid.to_dataset()
                    mdal_dataset = imod.util.spatial.mdal_compliant_ugrid2d(dataset)
                    mdal_dataset.to_netcdf(modeldirectory / pkg_path)
                else:
                    pkg.dataset.ugrid.to_netcdf(modeldirectory / pkg_path)
            else:
                pkg.to_netcdf(modeldirectory / pkg_path)

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

    @classmethod
    def model_id(cls) -> str:
        if cls._model_id is None:
            raise ValueError("Model id has not been set")
        return cls._model_id

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
        state_for_boundary :
        """
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

        Returns
        -------
        clipped : Modflow6Model
        """

        top, bottom, idomain = self.__get_domain_geometry()

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
                state_for_boundary=state_for_boundary,
            )

        return clipped

    def regrid_like(
        self, target_grid: GridDataArray, validate: bool = True
    ) -> "Modflow6Model":
        """
        Creates a model by regridding the packages of this model to another discretization.
        It regrids all the arrays in the package using the default regridding methods.
        At the moment only regridding to a different planar grid is supported, meaning
        ``target_grid`` has different ``"x"`` and ``"y"`` or different ``cell2d`` coords.

        Parameters
        ----------
        target_grid: xr.DataArray or xu.UgridDataArray
            a grid defined over the same discretization as the one we want to regrid the package to
        validate: bool
            set to true to validate the regridded packages

        Returns
        -------
        a model with similar packages to the input model, and with all the data-arrays regridded to another discretization,
        similar to the one used in input argument "target_grid"
        """

        return _regrid_like(self, target_grid, validate)
 
    def mask_all_packages(
        self,
        mask: GridDataArray,
    ):
        """
        This function applies a mask to all packages in a model. The mask must
        be presented as an idomain-like integer array that has 0 (inactive) or
        -1 (vertical passthrough) values in filtered cells and 1 in active
        cells.
        Masking will overwrite idomain with the mask where the mask is 0 or -1.
        Where the mask is 1, the original value of idomain will be kept. Masking
        will update the packages accordingly, blanking their input where needed,
        and is therefore not a reversible operation. 
        
        Parameters
        ----------
        mask: xr.DataArray, xu.UgridDataArray of ints
            idomain-like integer array. 1 sets cells to active, 0 sets cells to inactive, 
            -1 sets cells to vertical passthrough
        """

        _mask_all_packages(self, mask)

    def purge_empty_packages(self, model_name: Optional[str] = "") -> None:
        """
        This function removes empty packages from the model.
        """
        empty_packages = [
            package_name for package_name, package in self.items() if package.is_empty()
        ]
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

