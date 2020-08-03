import collections
import os
import pathlib
import subprocess

import cftime
import jinja2
import numpy as np
import pandas as pd
import xarray as xr

import imod
from imod.wq import timeutil
from imod.wq.pkggroup import PackageGroups


def _relpath(path, to):
    # Wraps os.path.relpath
    try:
        return pathlib.Path(os.path.relpath(path, to))
    except ValueError:
        # Fails to switch between drives e.g.
        return pathlib.Path(os.path.abspath(path))


# This class allows only imod packages as values
class Model(collections.UserDict):
    def __setitem__(self, key, value):
        # TODO: raise ValueError on setting certain duplicates
        # e.g. two solvers
        if self.check == "eager":
            value._pkgcheck()
        super(__class__, self).__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def visualize(
        self,
        directory,
        cmap="viridis",
        overlays=[],
        quantile_colorscale=True,
        figsize=(8, 8),
    ):
        directory = pathlib.Path(directory)

        for key, value in self.items():
            if value._pkg_id == "well":
                pass
                # grouped = value.groupby(["x", "y"])
                # x = grouped["x"].first()
                # y = grouped["y"].first()
            # Select appropriate datasets
            if "x" in value.dims and "y" in value.dims:
                for varname in value.data_vars:
                    da = value[varname]
                    if "x" in da.dims and "y" in da.dims:
                        if da.isnull().all():
                            continue

                        imod.visualize.spatial.imshow_topview(
                            da=da,
                            name=varname,
                            directory=directory / key,
                            cmap=cmap,
                            overlays=overlays,
                            quantile_colorscale=quantile_colorscale,
                            figsize=figsize,
                        )

    def sel(self, **dimensions):
        selmodel = type(self)(self.modelname, self.check)
        for pkgname, pkg in self.items():
            sel_dims = {k: v for k, v in dimensions.items() if k in pkg}
            if len(sel_dims) == 0:
                selmodel[pkgname] = pkg
            else:
                selmodel[pkgname] = pkg.loc[sel_dims]
        return selmodel

    def isel(self, **dimensions):
        selmodel = type(self)(self.modelname, self.check)
        for pkgname, pkg in self.items():
            sel_dims = {k: v for k, v in dimensions.items() if k in pkg}
            if len(sel_dims) == 0:
                selmodel[pkgname] = pkg
            else:
                selmodel[pkgname] = pkg[sel_dims]
        return selmodel


class SeawatModel(Model):
    """
    Attributes
    ----------
    modelname : str
    check : str, optional
        When to perform model checks {None, "defer", "eager"}.
        Defaults to "defer".

    Examples
    --------
    
    >>> m = SeawatModel("example")
    >>> m["riv"] = River(...)
    >>> # ...etc.
    >>> m.time_discretization(endtime)
    >>> m.write()
    """

    # These templates end up here since they require global information
    # from more than one package
    _PACKAGE_GROUPS = PackageGroups

    _gen_template = jinja2.Template(
        "[gen]\n"
        "    runtype = SEAWAT\n"
        "    modelname = {{modelname}}\n"
        "    writehelp = {{writehelp}}\n"
        "    result_dir = {{result_dir}}\n"
        "    packages = {{package_set|join(', ')}}\n"
        "    coord_xll = {{xmin}}\n"
        "    coord_yll = {{ymin}}\n"
        "    start_year = {{start_date[:4]}}\n"
        "    start_month = {{start_date[4:6]}}\n"
        "    start_day = {{start_date[6:8]}}\n"
        "    start_hour = {{start_date[8:10]}}\n"
        "    start_minute = {{start_date[10:12]}}\n"
    )

    def __init__(self, modelname, check="defer"):
        super(__class__, self).__init__()
        self.modelname = modelname
        self.check = check

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
                    has_group.update([pkg_id])

        package_groups = []
        for pkg_id, group in groups.items():
            # Create PackageGroup for every package
            # RiverGroup for rivers, DrainageGroup for drainage, etc.
            package_groups.append(self._PACKAGE_GROUPS[pkg_id].value(**group))

        return package_groups

    def _hastime(self, pkg):
        return (pkg._pkg_id == "wel" and "time" in pkg) or ("time" in pkg.coords)

    def _use_cftime(self):
        """
        Also checks if datetime types are homogeneous across packages.
        """
        types = []
        for pkg in self.values():
            if self._hastime(pkg):
                types.append(type(np.atleast_1d(pkg["time"].values)[0]))

        # Types will be empty if there's no time dependent input
        set_of_types = set(types)
        if len(set_of_types) == 0:
            return None
        else:  # there is time dependent input
            if not len(set_of_types) == 1:
                raise ValueError(
                    f"Multiple datetime types detected: {set_of_types}. "
                    "Use either cftime or numpy.datetime64[ns]."
                )
            # Since we compare types and not instances, we use issubclass
            if issubclass(types[0], cftime.datetime):
                return True
            elif issubclass(types[0], np.datetime64):
                return False
            else:
                raise ValueError("Use either cftime or numpy.datetime64[ns].")

    def time_discretization(self, times):
        """
        Collect all unique times

        Parameters
        ----------
        times : str, datetime; or iterable of str, datetimes.
            Times to add to the time discretization. At least one single time
            should be given, which will be used as the ending time of the
            simulation.

        Examples
        --------
        Add a single time:

        >>> m.time_discretization("2001-01-01")

        Add a daterange:

        >>> m.time_discretization(pd.daterange("2000-01-01", "2001-01-01"))
        
        Add a list of times:

        >>> m.time_discretization(["2000-01-01", "2001-01-01"])

        """

        # Make sure it's an iterable
        if not isinstance(times, (np.ndarray, list, tuple, pd.DatetimeIndex)):
            times = [times]

        # Loop through all packages, check if cftime is required.
        self.use_cftime = self._use_cftime()
        # use_cftime is None if you no datetimes are present in packages
        # use_cftime is False if np.datetimes present in packages
        # use_cftime is True if cftime.datetime present in packages
        for time in times:
            if issubclass(type(time), cftime.datetime):
                if self.use_cftime is None:
                    self.use_cftime = True
                if self.use_cftime is False:
                    raise ValueError(
                        "Use either cftime or numpy.datetime64[ns]. "
                        f"Received: {type(time)}."
                    )
        if self.use_cftime is None:
            self.use_cftime = False

        times = [timeutil.to_datetime(time, self.use_cftime) for time in times]
        first_times = {}  # first time per package
        for key, pkg in self.items():
            if self._hastime(pkg):
                pkgtimes = list(pkg["time"].values)
                first_times[key] = sorted(pkgtimes)[0]
                for var in pkg.data_vars:
                    if "timemap" in pkg[var].attrs:
                        timemap_times = list(pkg[var].attrs["timemap"].keys())
                        pkgtimes.extend(timemap_times)
                times.extend(pkgtimes)

        # np.unique also sorts
        times = np.unique(np.hstack(times))

        # Check if every transient package commences at the same time.
        for key, first_time in first_times.items():
            time0 = times[0]
            if first_time != time0:
                raise ValueError(
                    f"Package {key} does not have a value specified for the "
                    f"first time: {time0}. Every input must be present in the "
                    "first stress period. Values are only filled forward in "
                    "time."
                )

        duration = timeutil.timestep_duration(times, self.use_cftime)
        # Generate time discretization, just rely on default arguments
        # Probably won't be used that much anyway?
        timestep_duration = xr.DataArray(
            duration, coords={"time": np.array(times)[:-1]}, dims=("time",)
        )
        self["time_discretization"] = imod.wq.TimeDiscretization(
            timestep_duration=timestep_duration
        )

    def _render_gen(self, modelname, globaltimes, writehelp, result_dir):
        package_set = set(
            [pkg._pkg_id for pkg in self.values() if pkg._pkg_id not in ("tvc", "mal")]
        )
        package_set.update(("btn", "ssm"))
        package_set = sorted(package_set)
        baskey = self._get_pkgkey("bas6")
        bas = self[baskey]
        _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(bas["ibound"])

        if not self.use_cftime:
            start_time = pd.to_datetime(globaltimes[0])
        else:
            start_time = globaltimes[0]

        start_date = start_time.strftime("%Y%m%d%H%M%S")

        d = {}
        d["modelname"] = modelname
        d["writehelp"] = writehelp
        d["result_dir"] = result_dir
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
            raise ValueError(f"No {key} package provided.")
        return self[pkgkey]._render(
            directory=directory / pkgkey, globaltimes=globaltimes
        )

    def _render_dis(self, directory, globaltimes):
        baskey = self._get_pkgkey("bas6")
        diskey = self._get_pkgkey("dis")
        lpfkey = self._get_pkgkey("lpf")
        bas_content = self[baskey]._render_dis(directory=directory.joinpath(baskey))
        dis_content = self[diskey]._render(globaltimes=globaltimes)
        return bas_content + dis_content

    def _render_groups(self, directory, globaltimes):
        baskey = self._get_pkgkey("bas6")
        nlayer, nrow, ncol = self[baskey]["ibound"].shape
        package_groups = self._group()
        content = "\n\n".join(
            [
                group.render(directory, globaltimes, nlayer, nrow, ncol)
                for group in package_groups
            ]
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
        self[btnkey]["thickness"] = self[baskey].thickness()

        if btnkey is None:
            raise ValueError("No BasicTransport package provided.")
        btn_content = self[btnkey]._render(directory=directory.joinpath(btnkey))
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
            return self[pkstkey]._render(directory=directory / pkstkey)

    def _render_ssm_rch_mal_tvc(self, directory, globaltimes):
        out = ""
        for key, pkg in self.items():
            if pkg._pkg_id in ("rch", "mal", "tvc"):
                out += pkg._render_ssm(
                    directory=directory / key, globaltimes=globaltimes
                )
        return out

    def _bas_btn_rch_evt_mal_tvc_sinkssources(self):
        baskey = self._get_pkgkey("bas6")
        btnkey = self._get_pkgkey("btn")
        ibound = self[baskey]["ibound"]
        icbund = self[btnkey]["icbund"]
        n_extra = int(((ibound < 0) | (icbund < 0)).sum())

        nlayer, nrow, ncol = ibound.shape
        for key in ("rch", "evt", "mal", "tvc"):
            pkgkey = self._get_pkgkey(key)
            if pkgkey is not None:
                pkg = self[pkgkey]
                if key in ("rch", "evt"):
                    pkg._set_ssm_layers(ibound)
                    n_extra += pkg._ssm_layers.size * nrow * ncol
                elif key in ("mal", "tvc"):
                    _ = pkg._max_active_n("concentration", nlayer, nrow, ncol)
                    n_extra += pkg._ssm_cellcount

        return n_extra

    def render(self, directory, result_dir, writehelp):
        """
        Render the runfile as a string, package by package.
        """
        diskey = self._get_pkgkey("dis")
        globaltimes = self[diskey]["time"].values

        content = []
        content.append(
            self._render_gen(
                modelname=self.modelname,
                globaltimes=globaltimes,
                writehelp=writehelp,
                result_dir=result_dir,
            )
        )
        content.append(self._render_dis(directory=directory, globaltimes=globaltimes))
        # Modflow
        for key in ("bas6", "oc", "lpf"):
            content.append(
                self._render_pkg(key=key, directory=directory, globaltimes=globaltimes)
            )

        # multi-system package group: chd, drn, ghb, riv, wel
        modflowcontent, ssm_content, n_sinkssources = self._render_groups(
            directory=directory, globaltimes=globaltimes
        )
        # Add recharge to sinks and sources
        n_sinkssources += self._bas_btn_rch_evt_mal_tvc_sinkssources()
        # Add recharge, mass loading, time varying constant concentration to ssm_content
        ssm_content += self._render_ssm_rch_mal_tvc(
            directory=directory, globaltimes=globaltimes
        )

        # Wrap up modflow part
        content.append(modflowcontent)
        content.append(self._render_flowsolver(directory=directory))

        # MT3D and Seawat settings
        # Make an estimate of the required number of particles.
        advkey = self._get_pkgkey("adv")
        baskey = self._get_pkgkey("bas6")
        if isinstance(self[advkey], (imod.wq.AdvectionMOC, imod.wq.AdvectionHybridMOC)):
            ibound = self[baskey]["ibound"]
            cell_max_nparticles = self[advkey]["cell_max_nparticles"]
            self[advkey]["max_nparticles"] = int(
                np.product(ibound.shape) * 0.5 * cell_max_nparticles
            )

        content.append(self._render_btn(directory=directory, globaltimes=globaltimes))
        for key in ("vdf", "adv", "dsp"):
            content.append(
                self._render_pkg(key=key, directory=directory, globaltimes=globaltimes)
            )
        ssm_content = f"[ssm]\n    mxss = {n_sinkssources}" + ssm_content

        content.append(ssm_content)
        content.append(self._render_transportsolver(directory=directory))

        return "\n\n".join(content)

    def _set_caching_packages(self, reldir):
        # TODO:
        # Basically every package should rely on bas for checks via ibound
        # basic domain checking, etc.
        # baskey = self._get_pkgkey("bas6")
        # So far, only slv does? Others only depend on themselves.
        for pkgname, pkg in self.items():
            if hasattr(pkg, "_filehashself"):
                # Clear _outputfiles in case of repeat writes
                pkg._outputfiles = []
                pkg._filehashes[pkgname] = pkg._filehashself
                pkg._reldir = reldir
        # If more complex dependencies do show up, probably push methods down
        # to the individual packages.

    def write(
        self, directory=pathlib.Path("."), result_dir=None, resultdir_is_workdir=False
    ):
        """
        Writes model input files.

        Parameters
        ----------
        directory : str, pathlib.Path
            Directory into which the model input will be written. The model
            input will be written into a directory called modelname.
        result_dir : str, pathlib.Path
            Path to directory in which output will be written when running the
            model. Is written as the value of the ``result_dir`` key in the
            runfile.

            See the examples.
        resultdir_is_workdir: boolean, optional
            Wether the set all input paths in the runfile relative to the output
            directory. Because iMOD-wq generates a number of files in its working
            directory, it may be advantageous to set the working directory to
            a different path than the runfile location.

        Returns
        -------
        None

        Examples
        --------
        Say we wish to write the model input to a file called input, and we
        desire that when running the model, the results end up in a directory
        called output. We may run:

        >>> model.write(directory="input", result_dir="output")

        And in the runfile, a value of ``../../output`` will be written for
        result_dir.
        """
        # Coerce to pathlib.Path
        directory = pathlib.Path(directory)
        if result_dir is None:
            result_dir = pathlib.Path("results")
        else:
            result_dir = pathlib.Path(result_dir)

        # Create directories if necessary
        directory.mkdir(exist_ok=True, parents=True)
        result_dir.mkdir(exist_ok=True, parents=True)
        runfilepath = directory / f"{self.modelname}.run"
        results_runfilepath = result_dir / f"{self.modelname}.run"

        # Where will the model run?
        # Default is inputdir, next to runfile:
        # in that case, resultdir is relative to inputdir
        # If resultdir_is_workdir, inputdir is relative to resultdir
        # render_dir is the inputdir that is printed in the runfile.
        # result_dir is the resultdir that is printed in the runfile.
        # caching_reldir is from where to check for files. This location
        # is the same as the eventual model working dir.
        if resultdir_is_workdir:
            caching_reldir = result_dir
            if not directory.is_absolute():
                render_dir = _relpath(directory, result_dir)
            else:
                render_dir = directory
            result_dir = pathlib.Path(".")
        else:
            caching_reldir = directory
            render_dir = pathlib.Path(".")
            if not result_dir.is_absolute():
                result_dir = _relpath(result_dir, directory)

        # Check if any caching packages are present, and set necessary states.
        self._set_caching_packages(caching_reldir)

        if not self.check is None:
            self.package_check()

        runfile_content = self.render(
            directory=render_dir, result_dir=result_dir, writehelp=False
        )

        # Start writing
        # Write the runfile
        with open(runfilepath, "w") as f:
            f.write(runfile_content)
        # Also write the runfile in the workdir
        if resultdir_is_workdir:
            with open(results_runfilepath, "w") as f:
                f.write(runfile_content)

        # Write all IDFs and IPFs
        for pkgname, pkg in self.items():
            if "x" in pkg.coords and "y" in pkg.coords or pkg._pkg_id == "wel":
                try:
                    pkg.save(directory=directory / pkgname)
                except Exception as error:
                    raise RuntimeError(
                        f"An error occured during saving of package: {pkgname}."
                    ) from error

    def package_check(self):
        baskey = self._get_pkgkey("bas6")
        if baskey is not None:
            ibound = self[baskey]["ibound"]
            top = self[baskey]["top"]
            bottom = self[baskey]["bottom"]
        else:
            ibound = None
            top = None
            bottom = None

        for pkg in self.values():
            pkg._pkgcheck(ibound=ibound)
