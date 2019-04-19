"""
Contains an imodseawat model object
"""
import collections
import pathlib

import cftime
import jinja2
import numpy as np
import pandas as pd
import xarray as xr

import imod.wq
from imod.io import util
from imod.wq.pkggroup import PackageGroups


def _to_datetime(time):
    """
    Check whether time is cftime object, else convert to datetime64 series.
    
    cftime currently has no pd.to_datetime equivalent:
    a method that accepts a lot of different input types.
    
    Parameters
    ----------
    time : cftime object or datetime-like scalar
    """
    if isinstance(time, cftime.datetime):
        return time
    else:
        return pd.to_datetime(time)


def _timestep_duration(times):
    """
    Generates dictionary containing stress period time discretization data.

    Parameters
    ----------
    times : np.array
        Array containing containing time in a datetime-like format
    
    Returns
    -------
    collections.OrderedDict
        Dictionary with dates as strings for keys,
        stress period duration (in days) as values.
    """
    times = sorted([_to_datetime(t) for t in times])

    timestep_duration = []
    for start, end in zip(times[:-1], times[1:]):
        timedelta = end - start
        duration = timedelta.days + timedelta.seconds / 86400.0
        timestep_duration.append(duration)
    return times, timestep_duration


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
        "    modelname = {{modelname}}\n"
        "    writehelp = {{writehelp}}\n"
        "    result_dir = {{modelname}}\n"
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

    def time_discretization(self, endtime):
        """
        Collect all unique times
        """
        # TODO: check for cftime, force all to cftime if necessary
        times = set()  # set only allows for unique values
        for ds in self.values():
            if "time" in ds.coords:
                times.update(ds.coords["time"].values)
        # TODO: check that endtime is later than all other times.
        times.update((endtime,))
        times, duration = _timestep_duration(times)
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
        package_set.update(("btn",))
        package_set = sorted(package_set)
        baskey = self._get_pkgkey("bas")
        bas = self[baskey]
        _, xmin, xmax, _, ymin, ymax = util.spatial_reference(bas["ibound"])
        start_date = _to_datetime(globaltimes[0]).strftime("%Y%m%d%H%M%S")

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
        key = self._get_pkgkey(key)
        if key is None:
            # Maybe do enum look for full package name?
            raise ValueError(f"No {key} package provided.")
        return self[key]._render(directory=directory, globaltimes=globaltimes)

    def _render_dis(self, directory, globaltimes):
        baskey = self._get_pkgkey("bas")
        diskey = self._get_pkgkey("dis")
        bas_content = self[baskey]._render_dis(directory=directory)
        dis_content = self[diskey]._render(globaltimes=globaltimes)
        return bas_content + dis_content

    def _render_groups(self, directory, globaltimes):
        package_groups = self._group()
        content = "".join(
            [group.render(directory, globaltimes) for group in package_groups]
        )
        ssm_content = "".join(
            [group.render_ssm(directory, globaltimes) for group in package_groups]
        )
        # TODO: do this in a single pass, combined with _n_max_active for modflow part?
        n_sinkssources = sum([group.max_n_sinkssources() for group in package_groups])
        ssm_content = f"[ssm]\n    mxss = {n_sinkssources}\n" + ssm_content
        return content, ssm_content

    def _render_flowsolver(self):
        pcgkey = self._get_pkgkey("pcg")
        pksfkey = self._get_pkgkey("pksf")
        if pcgkey and pksfkey:
            raise ValueError("pcg and pksf solver both provided. Provide only one.")
        if not pcgkey and not pksfkey:
            raise ValueError("No flow solver provided")
        if pcgkey:
            return self[pcgkey]._render()
        else:
            return self[pksfkey]._render()

    def _render_btn(self, directory, globaltimes):
        baskey = self._get_pkgkey("bas")
        btnkey = self._get_pkgkey("btn")
        diskey = self._get_pkgkey("dis")
        thickness = self[baskey].thickness()

        if btnkey is None:
            raise ValueError("No BasicTransport package provided.")
        btn_content = self[btnkey]._render(directory=directory, thickness=thickness)
        dis_content = self[diskey]._render_btn(globaltimes=globaltimes)
        return btn_content + dis_content

    def _render_transportsolver(self):
        gcgkey = self._get_pkgkey("gcg")
        pkstkey = self._get_pkgkey("pksf")
        if gcgkey and pkstkey:
            raise ValueError("gcg and pkst solver both provided. Provide only one.")
        if not gcgkey and not pkstkey:
            raise ValueError("No transport solver provided")
        if gcgkey:
            return self[gcgkey]._render()
        else:
            return self[pkstkey]._render()

    def render(self, writehelp=False):
        """
        Render the runfile as a string, package by package.
        """
        diskey = self._get_pkgkey("dis")
        globaltimes = self[diskey]["time"].values
        directory = pathlib.Path(self.modelname)

        modflowcontent, ssmcontent = self._render_groups(
            directory=directory, globaltimes=globaltimes
        )

        content = []
        content.append(
            self._render_gen(
                modelname=self.modelname, globaltimes=globaltimes, writehelp=writehelp
            )
        )
        content.append(self._render_dis(directory=directory, globaltimes=globaltimes))
        # Modflow
        for key in ("bas", "oc", "lpf", "rch"):
            content.append(
                self._render_pkg(key=key, directory=directory, globaltimes=globaltimes)
            )
        content.append(modflowcontent)
        content.append(self._render_flowsolver())

        # MT3D and Seawat
        content.append(self._render_btn(directory=directory, globaltimes=globaltimes))
        for key in ("vdf", "adv", "dsp"):
            self._render_pkg(key=key, directory=directory, globaltimes=globaltimes)
        content.append(ssmcontent)
        content.append(self._render_transportsolver())

        return "\n".join(content)

    def save(self, directory):
        for ds in self.values():
            if isinstance(ds, imod.wq.Well):
                # TODO: implement
                raise NotImplementedError
            else:
                for name, da in ds.data_vars.items():
                    if "y" in da.coords and "x" in da.coords:
                        imod.io.idf.save(directory, da)

    def write(self):
        # TODO: just write to an arbitrary directory
        runfile_content = self.render()
        runfilepath = f"{self.modelname}.run"
        # Write the runfile
        with open(runfilepath, "w") as f:
            f.write(runfile_content)
        # Write all IDFs and IPFs
        self.save(self["modelname"])
