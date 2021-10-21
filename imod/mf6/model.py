import collections
import pathlib

import cftime
import jinja2
import numpy as np

from imod.mf6 import qgs_util


class Model(collections.UserDict):
    def __setitem__(self, key, value):
        # TODO: Add packagecheck
        super(__class__, self).__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class GroundwaterFlowModel(Model):
    """
    Contains data and writes consistent model input files
    """

    _pkg_id = "model"

    def _initialize_template(self):
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        self._template = env.get_template("gwf-nam.j2")

    def __init__(self, newton=False, under_relaxation=False):
        super(__class__, self).__init__()
        self.newton = newton
        self.under_relaxation = under_relaxation
        self._initialize_template()

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

    def render(self, modelname):
        """Render model namefile"""
        dir_for_render = pathlib.Path(modelname)
        d = {"newton": self.newton, "under_relaxation": self.under_relaxation}
        packages = []
        for pkgname, pkg in self.items():
            # Add the six to the package id
            pkg_id = pkg._pkg_id
            key = f"{pkg_id}6"
            path = dir_for_render / f"{pkgname}.{pkg_id}"
            packages.append((key, path.as_posix()))
        d["packages"] = packages
        return self._template.render(d)

    def write(self, wdir, modelname, globaltimes):
        """
        Write model namefile
        Write packages
        """

        wdir = pathlib.Path(wdir)

        modeldirectory = wdir / modelname
        modeldirectory.mkdir(exist_ok=True, parents=True)

        # write model namefile
        namefile_content = self.render(modelname)
        namefile_path = modeldirectory / f"{modelname}.nam"
        with open(namefile_path, "w") as f:
            f.write(namefile_content)

        # write package contents
        for pkgname, pkg in self.items():
            pkg.write(modeldirectory, pkgname, globaltimes)

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
