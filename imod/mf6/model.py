import abc
import collections
import inspect
import pathlib
from copy import deepcopy
from pathlib import Path

import cftime
import jinja2
import numpy as np
import tomli
import tomli_w
import xugrid as xu
from jinja2 import Template

import imod
from imod.mf6 import qgs_util
from imod.mf6.pkgbase import Package
from imod.mf6.statusinfo import NestedStatusInfo, StatusInfo, StatusInfoBase
from imod.mf6.validation import pkg_errors_to_status_info
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError


def initialize_template(name: str) -> Template:
    loader = jinja2.PackageLoader("imod", "templates/mf6")
    env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
    return env.get_template(name)


class Modflow6Model(collections.UserDict, abc.ABC):
    _mandatory_packages = None
    _model_id = None

    def __init__(self, **kwargs):
        collections.UserDict.__init__(self)
        for k, v in kwargs.items():
            self[k] = v

        self._options = {}
        self._template = None

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

    def __get_diskey(self):
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

    def render(self, modelname: str):
        dir_for_render = pathlib.Path(modelname)
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

    def _validate(self, model_name: str = "") -> StatusInfoBase:
        try:
            diskey = self.__get_diskey()
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

    def write(
        self, modelname, globaltimes, write_context: WriteContext
    ) -> StatusInfoBase:
        """
        Write model namefile
        Write packages
        """

        workdir = write_context.get_simulation_directory()
        modeldirectory = workdir / modelname
        Path(modeldirectory).mkdir(exist_ok=True, parents=True)
        if write_context.is_validate():
            model_status_info = self._validate(modelname)
            if model_status_info.has_errors():
                return model_status_info

        # write model namefile
        namefile_content = self.render(modelname)
        namefile_path = modeldirectory / f"{modelname}.nam"
        with open(namefile_path, "w") as f:
            f.write(namefile_content)

        # write package contents
        write_context.output_directory = modeldirectory
        for pkg_name, pkg in self.items():
            try:
                pkg.write(
                    pkgname=pkg_name,
                    globaltimes=globaltimes,
                    write_context=write_context,
                )
            except Exception as e:
                raise type(e)(f"{e}\nError occured while writing {pkg_name}")

        return NestedStatusInfo(modelname)

    def dump(
        self, directory, modelname, validate: bool = True, mdal_compliant: bool = False
    ):
        modeldirectory = pathlib.Path(directory) / modelname
        modeldirectory.mkdir(exist_ok=True, parents=True)
        if validate:
            statusinfo = self._validate()
            if statusinfo.has_errors():
                raise ValidationError(statusinfo.to_string())

        toml_content = collections.defaultdict(dict)
        for pkgname, pkg in self.items():
            pkg_path = f"{pkgname}.nc"
            toml_content[type(pkg).__name__][pkgname] = pkg_path
            dataset = pkg.dataset
            if isinstance(dataset, xu.UgridDataset):
                if mdal_compliant:
                    dataset = pkg.dataset.ugrid.to_dataset()
                    mdal_dataset = imod.util.mdal_compliant_ugrid2d(dataset)
                    mdal_dataset.to_netcdf(modeldirectory / pkg_path)
                else:
                    pkg.dataset.ugrid.to_netcdf(modeldirectory / pkg_path)
            else:
                pkg.dataset.to_netcdf(modeldirectory / pkg_path)

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
        x_min: optional, float
        y_max: optional, float
        y_max: optional, float

        Returns
        -------
        clipped : Modflow6Model
        """
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
            )
        return clipped


class GroundwaterFlowModel(Modflow6Model):
    _mandatory_packages = ("npf", "ic", "oc", "sto")
    _model_id = "gwf6"

    def __init__(
        self,
        listing_file: str = None,
        print_input: bool = False,
        print_flows: bool = False,
        save_flows: bool = False,
        newton: bool = False,
        under_relaxation: bool = False,
    ):
        super().__init__()
        self._options = {
            "listing_file": listing_file,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "newton": newton,
            "under_relaxation": under_relaxation,
        }
        self._template = initialize_template("gwf-nam.j2")

    def write_qgis_project(self, directory, crs, aggregate_layers=False):
        """
        Write qgis projectfile and accompanying netcdf files that can be read in qgis.

        Parameters
        ----------
        directory : Path
            directory of qgis project

        crs : str, int,
            anything that can be converted to a pyproj.crs.CRS

        aggregate_layers : Optional, bool
            If True, aggregate layers by taking the mean, i.e. ds.mean(dim="layer")

        """
        ext = ".qgs"

        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        pkgnames = [
            pkgname
            for pkgname, pkg in self.items()
            if all(i in pkg.dataset.dims for i in ["x", "y"])
        ]

        data_paths = []
        data_vars_ls = []
        for pkgname in pkgnames:
            pkg = self[pkgname].rio.write_crs(crs)
            data_path = pkg._netcdf_path(directory, pkgname)
            data_path = "./" + data_path.relative_to(directory).as_posix()
            data_paths.append(data_path)
            # FUTURE: MDAL has matured enough that we do not necessarily
            #           have to write seperate netcdfs anymore
            data_vars_ls.append(
                pkg.write_netcdf(directory, pkgname, aggregate_layers=aggregate_layers)
            )

        qgs_tree = qgs_util._create_qgis_tree(
            self, pkgnames, data_paths, data_vars_ls, crs
        )
        qgs_util._write_qgis_projectfile(qgs_tree, directory / ("qgis_proj" + ext))


class GroundwaterTransportModel(Modflow6Model):
    """
    The GroundwaterTransportModel (GWT) simulates transport of a single solute
    species flowing in groundwater.
    """

    _mandatory_packages = ("mst", "dsp", "oc", "ic")
    _model_id = "gwt6"

    def __init__(
        self,
        listing_file: str = None,
        print_input: bool = False,
        print_flows: bool = False,
        save_flows: bool = False,
    ):
        super().__init__()
        self._options = {
            "listing_file": listing_file,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
        }

        self._template = initialize_template("gwt-nam.j2")
