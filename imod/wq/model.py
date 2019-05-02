import collections
import pathlib

import cftime
import imod
import jinja2
import numpy as np
import pandas as pd
import xarray as xr
from imod.wq import timeutil
from imod.wq.pkggroup import PackageGroups


# This class allows only imod packages as values
class Model(collections.UserDict):
    pass


#    def __setitem__(self, key, value):
#        # TODO: raise ValueError on setting certain duplicates
#        # e.g. two solvers
#        if not hasattr(value, "_pkg_id"):
#            raise ValueError("The value to set is not an imod package")
#        dict.__setitem__(self, key, value)
#
#    def update(self, *args, **kwargs):
#        for k, v in dict(*args, **kwargs).items():
#            self[k] = v


class SeawatModel(Model):
    """
    Examples
    --------
    m = SeawatModel("example")
    m["riv"] = River(...)
    ...etc.
    m.time_discretization(endtime)
    m.write()
    """

    # These templates end up here since they require global information
    # from more than one package
    _PACKAGE_GROUPS = PackageGroups

    _gen_template = jinja2.Template(
        "[gen]\n"
        "    runtype = SEAWAT\n"
        "    modelname = {{modelname}}\n"
        "    writehelp = {{writehelp}}\n"
        "    result_dir = results\n"
        "    packages = {{package_set|join(', ')}}\n"
        "    coord_xll = {{xmin}}\n"
        "    coord_yll = {{ymin}}\n"
        "    start_year = {{start_date[:4]}}\n"
        "    start_month = {{start_date[4:6]}}\n"
        "    start_day = {{start_date[6:8]}}\n"
    )

    def __init__(self, modelname):
        super(__class__, self).__init__()
        self.modelname = modelname

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

    def _group(self):
        """
        Group multiple systems of a single package E.g. all river or drainage
        sub-systems
        """
        groups = {}
        has_group = set()
        groupable = set(self._PACKAGE_GROUPS.__members__.keys())
        for key, package in self.items():
            pkg_id = package._pkg_id
            if pkg_id in groupable:
                if pkg_id in has_group:  # already exists
                    groups[pkg_id][key] = package
                else:
                    groups[pkg_id] = {key: package}
                    has_group.update(pkg_id)

        package_groups = []
        for pkg_id, group in groups.items():
            # Create PackageGroup for every package
            # RiverGroup for rivers, DrainageGroup for drainage, etc.
            package_groups.append(self._PACKAGE_GROUPS[pkg_id].value(**group))

        return package_groups

    def _use_cftime(self):
        """
        Also checks if datetime types are homogeneous across packages.
        """
        types = [type(pkg["time"].values[0]) for pkg in self.values()]
        if not len(set(types)) == 1:
            raise ValueError(
                "Multiple datetime types detected. "
                "Use either cftime or numpy.datetime64[ns]."
            )
        if isinstance(types[0], cftime.datetime):
            return True
        elif isinstance(types[0], np.datetime64):
            return False
        else:
            raise ValueError("Use either cftime or numpy.datetime64[ns].")

    def time_discretization(self, endtime, starttime=None):
        """
        Collect all unique times
        """
        use_cftime = self._use_cftime()

        times = []
        for pkg in self.values():
            if "time" in pkg.coords:
                times.append(pkg["time"].values)

        # TODO: check that endtime is later than all other times.
        times.append(timeutil.to_datetime(endtime, use_cftime))
        if starttime is not None:
            times.append(timeutil.to_datetime(starttime, use_cftime))

        # np.unique also sorts
        times = np.unique(np.hstack(times))

        duration = timeutil.timestep_duration(times, use_cftime)
        # Generate time discretization, just rely on default arguments
        # Probably won't be used that much anyway?
        timestep_duration = xr.DataArray(
            duration, coords={"time": np.array(times)[:-1]}, dims=("time",)
        )
        self["time_discretization"] = imod.wq.TimeDiscretization(
            timestep_duration=timestep_duration
        )

    def _render_gen(self, modelname, globaltimes, writehelp=False):
        package_set = set([pkg._pkg_id for pkg in self.values()])
        package_set.update(("btn", "ssm"))
        package_set = sorted(package_set)
        baskey = self._get_pkgkey("bas6")
        bas = self[baskey]
        _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(bas["ibound"])
        start_date = timeutil.to_datetime(globaltimes[0]).strftime("%Y%m%d%H%M%S")

        d = {}
        d["modelname"] = modelname
        d["writehelp"] = writehelp
        d["result_dir"] = modelname
        d["xmin"] = xmin
        d["xmax"] = xmax
        d["ymin"] = ymin
        d["ymax"] = ymax
        d["package_set"] = package_set
        d["start_date"] = start_date
        return self._gen_template.render(d)

    def _render_pkg(self, key, directory, globaltimes):
        """
        Rendering method for straightforward packages
        """
        # Get name of pkg, e.g. lookup "recharge" for rch _pkg_id
        pkgkey = self._get_pkgkey(key)
        if pkgkey is None:
            # Maybe do enum look for full package name?
            if key == "rch":  # since recharge is optional
                return ""
            else:
                raise ValueError(f"No {key} package provided.")
        return self[pkgkey]._render(
            directory=directory.joinpath(pkgkey), globaltimes=globaltimes
        )

    def _render_dis(self, directory, globaltimes):
        baskey = self._get_pkgkey("bas6")
        diskey = self._get_pkgkey("dis")
        bas_content = self[baskey]._render_dis(directory=directory.joinpath(baskey))
        dis_content = self[diskey]._render(globaltimes=globaltimes)
        return bas_content + dis_content

    def _render_groups(self, directory, globaltimes):
        baskey = self._get_pkgkey("bas6")
        nlayer = self[baskey]["ibound"].shape[0]
        package_groups = self._group()
        content = "\n\n".join(
            [group.render(directory, globaltimes, nlayer) for group in package_groups]
        )
        ssm_content = "".join(
            [group.render_ssm(directory, globaltimes) for group in package_groups]
        )

        # Calculate number of sinks and sources
        n_sinkssources = sum([group.max_n_sinkssources() for group in package_groups])
        return content, ssm_content, n_sinkssources

    def _render_flowsolver(self, directory):
        pcgkey = self._get_pkgkey("pcg")
        pksfkey = self._get_pkgkey("pksf")
        if pcgkey and pksfkey:
            raise ValueError("pcg and pksf solver both provided. Provide only one.")
        if not pcgkey and not pksfkey:
            raise ValueError("No flow solver provided")
        if pcgkey:
            return self[pcgkey]._render()
        else:
            baskey = self._get_pkgkey("bas6")
            self[pksfkey]._compute_load_balance_weight(self[baskey]["ibound"])
            return self[pksfkey]._render(directory=directory.joinpath(pksfkey))

    def _render_btn(self, directory, globaltimes):
        baskey = self._get_pkgkey("bas6")
        btnkey = self._get_pkgkey("btn")
        diskey = self._get_pkgkey("dis")
        thickness = self[baskey].thickness()

        if btnkey is None:
            raise ValueError("No BasicTransport package provided.")
        btn_content = self[btnkey]._render(
            directory=directory.joinpath(btnkey), thickness=thickness
        )
        dis_content = self[diskey]._render_btn(globaltimes=globaltimes)
        return btn_content + dis_content

    def _render_transportsolver(self, directory):
        gcgkey = self._get_pkgkey("gcg")
        pkstkey = self._get_pkgkey("pkst")
        if gcgkey and pkstkey:
            raise ValueError("gcg and pkst solver both provided. Provide only one.")
        if not gcgkey and not pkstkey:
            raise ValueError("No transport solver provided")
        if gcgkey:
            return self[gcgkey]._render()
        else:
            baskey = self._get_pkgkey("bas6")
            self[pkstkey]._compute_load_balance_weight(self[baskey]["ibound"])
            return self[pkstkey]._render(directory=directory.joinpath(pkstkey))

    def _render_ssm_rch(self, directory, globaltimes):
        rchkey = self._get_pkgkey("rch")
        if rchkey is not None:
            return self[rchkey]._render_ssm(
                directory=directory, globaltimes=globaltimes
            )
        else:
            return ""

    def _btn_rch_sinkssources(self):
        btnkey = self._get_pkgkey("btn")
        icbund = self[btnkey]["icbund"]
        n_extra = icbund.where(icbund == -1).count()

        rchkey = self._get_pkgkey("rch")
        if rchkey is not None:
            n_extra += self[rchkey]["rate"].count()

        return int(n_extra)

    def render(self, writehelp=False):
        """
        Render the runfile as a string, package by package.
        """
        diskey = self._get_pkgkey("dis")
        globaltimes = self[diskey]["time"].values
        # directory = pathlib.Path(self.modelname)
        directory = pathlib.Path(".")

        content = []
        content.append(
            self._render_gen(
                modelname=self.modelname, globaltimes=globaltimes, writehelp=writehelp
            )
        )
        content.append(self._render_dis(directory=directory, globaltimes=globaltimes))
        # Modflow
        for key in ("bas6", "oc", "lpf", "rch"):
            content.append(
                self._render_pkg(key=key, directory=directory, globaltimes=globaltimes)
            )

        # multi-system package group: chd, drn, ghb, riv, wel
        modflowcontent, ssm_content, n_sinkssources = self._render_groups(
            directory=directory, globaltimes=globaltimes
        )
        # Add recharge to ssm_content
        ssm_content += self._render_ssm_rch(
            directory=directory, globaltimes=globaltimes
        )
        # Add recharge to sinks and sources
        n_sinkssources += self._btn_rch_sinkssources()

        # Wrap up modflow part
        content.append(modflowcontent)
        content.append(self._render_flowsolver(directory=directory))

        # MT3D and Seawat settings
        content.append(self._render_btn(directory=directory, globaltimes=globaltimes))
        for key in ("vdf", "adv", "dsp"):
            content.append(
                self._render_pkg(key=key, directory=directory, globaltimes=globaltimes)
            )
        ssm_content = f"[ssm]\n    mxss = {n_sinkssources}" + ssm_content

        content.append(ssm_content)
        content.append(self._render_transportsolver(directory=directory))

        return "\n\n".join(content)

    def write(self, directory="."):
        runfile_content = self.render()
        directory = pathlib.Path(directory).joinpath(self.modelname)
        directory.mkdir(exist_ok=True, parents=True)
        runfilepath = directory.joinpath(f"{self.modelname}.run")
        # Write the runfile
        with open(runfilepath, "w") as f:
            f.write(runfile_content)
        # Write all IDFs and IPFs
        for pkgname, pkg in self.items():
            if "x" in pkg.coords and "y" in pkg.coords:
                pkg.save(directory=directory.joinpath(pkgname))
