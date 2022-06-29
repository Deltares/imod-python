import collections
import pathlib

import cftime
import jinja2
import numpy as np

from imod.mf6 import qgs_util


class Model(collections.UserDict):
    def __setitem__(self, key, value):
        # TODO: Add packagecheck
        super().__setitem__(key, value)

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
        super().__init__()
        self.newton = newton
        self.under_relaxation = under_relaxation
        self._initialize_template()

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
        for required in ["npf", "ic", "oc", "sto"]:
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
            packages.append((key, path.as_posix(), pkgname))
        d["packages"] = packages
        return self._template.render(d)

    def _check_nan_in_active_cell(self, modelkey: str):
        """Check if nan is present in active cells"""
        diskey = self._get_diskey()

        # "If the IDOMAIN value for a cell is 1 or greater, the cell exists in
        # the simulation"
        active = self[diskey]["idomain"] >= 1

        pkg_ids_to_check = ["npf", "ic", "sto"]
        pkgkeys_to_check = [self._get_pkgkey(pkg_id) for pkg_id in pkg_ids_to_check]
        pkgkeys_to_check.append(diskey)

        for pkgkey in pkgkeys_to_check:
            pkg = self[pkgkey]
            variables = pkg._get_vars_to_check()

            for var in variables:
                nan_in_active = np.isnan(pkg.dataset[var]) & active
                if nan_in_active.any():
                    pkgname = pkg.__class__.__name__
                    raise ValueError(
                        f"Detected value with np.nan in active domain of model "
                        f"{modelkey} in {pkgname} for variable: {var}."
                    )

    def _check_river_bottom_below_model_bottom(self, modelkey: str):
        """
        Check if river bottom not below model bottom. Modflow 6 throws an
        error if this occurs.
        """

        diskey = self._get_diskey()

        bottom = self[diskey].dataset["bottom"]

        rivkeys = [pkgname for pkgname, pkg in self.items() if pkg._pkg_id == "riv"]

        for rivkey in rivkeys:
            riv = self[rivkey]
            riv_below_bottom = riv.dataset["bottom_elevation"] < bottom
            if riv_below_bottom.any():
                raise ValueError(
                    f"River bottom below model bottom for pkg '{rivkey}' "
                    f"in model '{modelkey}'"
                )

    def _model_checks(self, modelkey: str):
        """
        Check model integrity (called before writing)
        """
        self._check_for_required_packages(modelkey)
        self._check_nan_in_active_cell(modelkey)
        self._check_river_bottom_below_model_bottom(modelkey)

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
