import abc
import collections
import pathlib

import cftime
import jinja2
import numpy as np

from imod.mf6 import qgs_util, ssm


class Modflow6Model(collections.UserDict, abc.ABC):

    _pkg_id = "model"

    def __setitem__(self, key, value):
        # TODO: Add packagecheck
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def _initialize_template(self):
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        self._template = env.get_template("gwf-nam.j2")

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
        return modeltimes

    def _render(self, modelname: str, **kwargs):
        dir_for_render = pathlib.Path(modelname)
        d = kwargs
        packages = []
        for pkgname, pkg in self.items():
            # Add the six to the package id
            pkg_id = pkg._pkg_id
            key = f"{pkg_id}6"
            path = dir_for_render / f"{pkgname}.{pkg_id}"
            packages.append((key, path.as_posix(), pkgname))
        d["packages"] = packages
        return self._template.render(d)

    def write(self, directory, modelname, globaltimes, binary=True):
        """
        Write model namefile
        Write packages
        """
        workdir = pathlib.Path(directory)
        modeldirectory = workdir / modelname
        modeldirectory.mkdir(exist_ok=True, parents=True)

        # write model namefile
        namefile_content = self.render(modelname)
        namefile_path = modeldirectory / f"{modelname}.nam"
        with open(namefile_path, "w") as f:
            f.write(namefile_content)

        # write package contents
        for pkgname, pkg in self.items():
            pkg.write(
                directory=modeldirectory,
                pkgname=pkgname,
                globaltimes=globaltimes,
                binary=binary,
            )

    def take_discretization_from_model(self, other):
        # Check for mandatory packages
        discretization_ids = ("dis", "disv", "disu")
        for key, package in other.items():
            if  package._pkg_id in discretization_ids:
                newkey = self.generate_unique_key(key)
                self[newkey]= package


    def generate_unique_key(self, base):
        if base is None or base == "":
            base = "a"
        if base in self.keys():
            base = self.generate_unique_key(base + "x")
        return base



class GroundwaterFlowModel(Modflow6Model):
    _mandatory_packages = ("npf", "ic", "oc", "sto")
    _model_type = "gwf6"

    def __init__(self, newton=False, under_relaxation=False):
        super().__init__()
        self.newton = newton
        self.under_relaxation = under_relaxation
        self._initialize_template()

    def render(self, modelname: str):
        """Render model namefile"""
        return self._render(
            modelname, newton=self.newton, under_relaxation=self.under_relaxation
        )

    def write_qgis_project(self, directory, crs, aggregate_layers=False):
        """
        Write qgis projectfile and accompanying netcdf files that can be read in qgis.

        Parameters
        ----------
        directory : Path
            directory of qgis project

        crs : str, int,
            anything that can be converted to a pyproj.crs.CRS

        filename : Optional, str
            name of qgis projectfile.

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

    _mandatory_packages = ("mst", "dsp", "oc", "ic")
    _model_type = "gwt6"

    def __init__(self, flow_model, state_variable_name):
        super().__init__()
        self._initialize_template()
        if flow_model is not None:
            self["ssm"] = ssm.Transport_Sink_Sources(flow_model, state_variable_name)

    def render(self, modelname: str):
        """Render model namefile"""
        return self._render(modelname)
