import collections
import os
import pathlib
import subprocess

import cftime
import jinja2
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
import xarray as xr
from xarray.core.utils import either_dict_or_kwargs

import imod
from imod.wq import timeutil
from imod.wq.pkggroup import PackageGroups
from imod.mf6 import qgs_util


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

    def __repr__(self):
        reprstr = f"{type(self)}\nAvailable packages:\n"
        pkgstr = [f"{pkgname} ({pkg._pkg_id})" for pkgname, pkg in self.items()]
        reprstr = f"{reprstr}{', '.join(sorted(pkgstr))}\n"

        times = self._yield_times()
        if len(times):
            times = sorted(np.unique(times))
            timestr = []
            for t in times:
                if isinstance(t, np.datetime64):
                    t = pd.to_datetime(t)
                timestr.append(f"{t:%Y-%m-%dT%H:%M:%S}")
            reprstr = f"{reprstr}Times present:\n{', '.join(timestr)}"
        return reprstr

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

    def sel(
        self, indexers=None, method=None, tolerance=None, drop=False, **indexers_kwargs
    ) -> "Model":
        """Returns a new Model with each array indexed by tick labels along the
        specified dimension(s).

        Indexers for this method should use labels instead of integers.

        The Model.sel method passes the .sel method down to each Package in
        Model.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by scalars,
            slices or arrays of tick labels. For dimensions with multi-index,
            the indexer may also be a dict-like object with keys matching index
            level names. If DataArrays are passed as indexers, xarray-style
            indexing will be carried out.
            Dimensions not present in a Package are ignored.
            One of indexers or indexers_kwargs must be provided.

        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for inexact matches:

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
        **indexers_kwarg : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Model
            A new Model with the same contents as this Model, except each
            variable and dimension in each Package is indexed by the appropriate
            indexers. If indexer DataArrays have coordinates that do not
            conflict with this object, then these coordinates will be attached.

        See Also
        --------
        xarray.Dataset.sel, Package.sel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        selmodel = type(self)(self.modelname, self.check)

        for pkgname, pkg in self.items():
            pkg_indexers = {k: v for k, v in indexers.items() if k in pkg}
            if len(pkg_indexers) == 0:
                selmodel[pkgname] = pkg
            else:
                selmodel[pkgname] = pkg.sel(
                    pkg_indexers, method=method, tolerance=tolerance, drop=drop
                )
        return selmodel

    def isel(
        self,
        indexers=None,
        drop=False,
        **indexers_kwargs,
    ) -> "Model":
        """Returns a new Model with each array indexed by tick labels along the
        specified dimension(s).

        This method selects values from each array using its `__getitem__`
        method, except this method does not require knowing the order of
        each array's dimensions.

        The Model.isel method passes the .isel method down to each Package in
        Model.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        **indexers_kwarg : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Model
            A new Model with the same contents as this Model, except each
            array and dimension of each Package is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this dataset, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.sel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        selmodel = type(self)(self.modelname, self.check)

        for pkgname, pkg in self.items():
            pkg_indexers = {k: v for k, v in indexers.items() if k in pkg}
            if len(pkg_indexers) == 0:
                selmodel[pkgname] = pkg
            else:
                selmodel[pkgname] = pkg.isel(pkg_indexers, drop=drop)
        return selmodel

    def _yield_times(self):
        """required by qgis_util"""
        modeltimes = []
        for pkg in self.values():
            if "time" in pkg.coords:
                modeltimes.append(pkg["time"].values)
        return modeltimes

    def to_netcdf(self, directory=".", pattern="{pkgname}.nc", **kwargs):
        """Convenience function to write all model packages
        to netcdf files.

        Parameters
        ----------
        directory : str, pathlib.Path
            Directory into which the different model packages will be written.

        pattern : str, optional.
            Pattern for filename of each package, in which `pkgname`
            signifies the package name. Default is `"{pkgname}.nc"`,
            so `model["river"]` would get written to `path / river.nc`.

        kwargs :
            Additional kwargs to be forwarded to `xarray.Dataset.to_netcdf`.
        """
        directory = pathlib.Path(directory)
        for pkgname, pkg in self.items():
            try:
                pkg.to_netcdf(directory / pattern.format(pkgname=pkgname), **kwargs)
            except:
                print(f"Package {pkgname} can not be written to NetCDF")
                raise

    def write_qgis_project(self, directory, crs, filename=None, aggregate_layers=False):
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
        if filename is None:
            filename = self.modelname

        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        pkgnames = [
            pkgname
            for pkgname, pkg in self.items()
            if all(i in pkg.dims for i in ["x", "y"])
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
        qgs_util._write_qgis_projectfile(qgs_tree, directory / (filename + ext))

    def _delete_empty_packages(self, verbose=False):
        to_del = []
        for pkg in self.keys():
            dv = list(self[pkg].data_vars)[0]
            if not self[pkg][dv].notnull().any().compute():
                if verbose:
                    print(f"Deleting package {pkg}, found no data in parameter {dv}")
                to_del.append(pkg)
        for pkg in to_del:
            del self[pkg]


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
        Collect all unique times from model packages and additional given
        ``times``. These unique times are used as stress periods in the model.
        All stress packages must have the same starting time.

        The time discretization in imod-python works as follows:

        * The datetimes of all packages you send in are always respected
        * Subsequently, the input data you use is always included fully as well
        * All times are treated as starting times for the stress: a stress is
          always applied until the next specified date
        * For this reason, a final time is required to determine the length of
          the last stress period
        * Additional times can be provided to force shorter stress periods &
          more detailed output
        * Every stress has to be defined on the first stress period (this is a
          modflow requirement)

        Or visually (every letter a date in the time axes)::

            recharge a - b - c - d - e - f
            river    g - - - - h - - - - j
            times    - - - - - - - - - - - i

            model    a - b - c h d - e - f i

        With the stress periods defined between these dates. I.e. the model
        times are the set of all times you include in the model.

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

    def _render_pkg(self, key, directory, globaltimes, nlayer):
        """
        Rendering method for straightforward packages
        """
        # Get name of pkg, e.g. lookup "recharge" for rch _pkg_id
        pkgkey = self._get_pkgkey(key)
        if pkgkey is None:
            # Maybe do enum look for full package name?
            if (key == "rch") or (key == "evt"):  # since recharge is optional
                return ""
            else:
                raise ValueError(f"No {key} package provided.")
        return self[pkgkey]._render(
            directory=directory / pkgkey, globaltimes=globaltimes, nlayer=nlayer
        )

    def _render_dis(self, directory, globaltimes, nlayer):
        baskey = self._get_pkgkey("bas6")
        diskey = self._get_pkgkey("dis")
        lpfkey = self._get_pkgkey("lpf")
        bas_content = self[baskey]._render_dis(
            directory=directory.joinpath(baskey), nlayer=nlayer
        )
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
            [
                group.render_ssm(directory, globaltimes, nlayer)
                for group in package_groups
            ]
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

    def _render_btn(self, directory, globaltimes, nlayer):
        baskey = self._get_pkgkey("bas6")
        btnkey = self._get_pkgkey("btn")
        diskey = self._get_pkgkey("dis")
        self[btnkey]["thickness"] = self[baskey].thickness()

        if btnkey is None:
            raise ValueError("No BasicTransport package provided.")
        btn_content = self[btnkey]._render(
            directory=directory.joinpath(btnkey), nlayer=nlayer
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
            return self[pkstkey]._render(directory=directory / pkstkey)

    def _render_ssm_rch_mal_tvc(self, directory, globaltimes, nlayer):
        out = ""
        for key, pkg in self.items():
            if pkg._pkg_id in ("rch", "mal", "tvc"):
                out += pkg._render_ssm(
                    directory=directory / key, globaltimes=globaltimes, nlayer=nlayer
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
        baskey = self._get_pkgkey("bas6")
        nlayer = self[baskey]["layer"].size

        content = []
        content.append(
            self._render_gen(
                modelname=self.modelname,
                globaltimes=globaltimes,
                writehelp=writehelp,
                result_dir=result_dir,
            )
        )
        content.append(
            self._render_dis(
                directory=directory, globaltimes=globaltimes, nlayer=nlayer
            )
        )
        # Modflow
        for key in ("bas6", "oc", "lpf", "rch", "evt"):
            content.append(
                self._render_pkg(
                    key=key, directory=directory, globaltimes=globaltimes, nlayer=nlayer
                )
            )

        # multi-system package group: chd, drn, ghb, riv, wel
        modflowcontent, ssm_content, n_sinkssources = self._render_groups(
            directory=directory, globaltimes=globaltimes
        )
        # Add recharge to sinks and sources
        n_sinkssources += self._bas_btn_rch_evt_mal_tvc_sinkssources()
        # Add recharge, mass loading, time varying constant concentration to ssm_content
        ssm_content += self._render_ssm_rch_mal_tvc(
            directory=directory, globaltimes=globaltimes, nlayer=nlayer
        )

        # Wrap up modflow part
        content.append(modflowcontent)
        content.append(self._render_flowsolver(directory=directory))

        # MT3D and Seawat settings
        # Make an estimate of the required number of particles.
        advkey = self._get_pkgkey("adv")
        if isinstance(self[advkey], (imod.wq.AdvectionMOC, imod.wq.AdvectionHybridMOC)):
            ibound = self[baskey]["ibound"]
            cell_max_nparticles = self[advkey]["cell_max_nparticles"]
            self[advkey]["max_nparticles"] = int(
                np.product(ibound.shape) * 0.5 * cell_max_nparticles
            )

        content.append(
            self._render_btn(
                directory=directory, globaltimes=globaltimes, nlayer=nlayer
            )
        )
        for key in ("vdf", "adv", "dsp"):
            content.append(
                self._render_pkg(
                    key=key, directory=directory, globaltimes=globaltimes, nlayer=nlayer
                )
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

        # Delete packages without data
        self._delete_empty_packages(verbose=True)

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

    def clip(
        self,
        extent,
        heads_boundary=None,
        concentration_boundary=None,
        delete_empty_pkg=False,
    ):
        """
        Method to clip the model to a certain `extent`. The spatial resolution
        of the clipped model is unchanged. Boundary conditions of clipped model
        can be derived from parent model calculation results and are applied
        along the edge of `extent` (CHD and TVC). Packages from parent that have
        no data within extent are optionally removed.

        Parameters
        ----------
        extent : tuple, geopandas.GeoDataFrame, xarray.DataArray
            Extent of the clipped model. Tuple must be in the form of
            (`xmin`,`xmax`,`ymin`,`ymax`). If a GeoDataFrame, all polygons are
            included in the model extent. If a DataArray, non-null/non-zero
            values are taken as the new extent.

        heads_boundary : xarray.DataArray, optional.
            Heads to be applied as a Constant Head boundary condition along the
            edge of the model extent. These heads are assumed to be derived from
            calculations with the parent model. Timestamp of boundary condition
            is shifted to correct for difference between 'end of period'
            timestamp of results and 'start of period' timestamp of boundary
            condition. If None (default), no constant heads boundary condition
            is applied.

        concentration_boundary : xarray.DataArray, optional.
            Concentration to be applied as a Time Varying Concentration boundary
            condition along the edge of the model extent. These concentrations
            can be derived from calculations with the parent model. Timestamp of
            boundary condition is shifted to correct for difference between 'end
            of period' timestamp of results and 'start of period' timestamp of
            boundary condition. If None (default), no time varying concentration
            boundary condition is applied.

            Note: the Time Varying Concentration boundary sets a constant
            concentration for the entire stress period, unlike the linearly
            varying Constant Head. This will inevitably cause a time shift in
            concentrations along the boundary.  This shift becomes more
            significant when stress periods are longer. If necessary, consider
            interpolating concentrations along the time axis, to reduce the
            length of stress periods (see examples).

        delete_empty_pkg : bool, optional.
            Set to True to delete packages that contain no data in the clipped
            model. Defaults to False.

        Examples
        --------
        Given a full model, clip a 1 x 1 km rectangular submodel without
        boundary conditions along its edge:

        >>> extent = (1000., 2000., 5000., 6000.)  # xmin, xmax, ymin, ymax
        >>> clipped = ml.clip(extent)

        Load heads and concentrations from full model results:

        >>> heads = imod.idf.open("head/head_*.idf")
        >>> conc = imod.idf.open("conc/conc_*.idf")
        >>> clipped = ml.clip(extent, heads, conc)

        Use a shape of a model area:

        >>> extent = geopandas.read_file("clipped_model_area.shp")
        >>> clipped = ml.clip(extent, heads, conc)

        Interpolate concentration results to annual results using
        xarray.interp(), to improve time resolution of concentration boundary:

        >>> conc = imod.idf.open("conc/conc_*.idf")
        >>> dates = pd.date_range(conc.time.values[0], conc.time.values[-1], freq="AS")
        >>> conc_interpolated = conc.load().interp(time=dates, method="linear")
        >>> clipped = ml.clip(extent, heads, conc_interpolated)
        """
        baskey = self._get_pkgkey("bas6")
        like = self[baskey]["ibound"].isel(layer=0).squeeze(drop=True)

        if isinstance(extent, (list, tuple)):
            xmin, xmax, ymin, ymax = extent
            if not ((xmin < xmax) & (ymin < ymax)):
                raise ValueError(
                    "Either xmin or ymin is equal to or larger than xmax or ymax. Correct order is xmin, xmax, ymin, ymax."
                )
            extent = xr.ones_like(like)
            extent = extent.where(
                (extent.x >= xmin)
                & (extent.x <= xmax)
                & (extent.y >= ymin)
                & (extent.y <= ymax),
                0,
            )
        elif isinstance(extent, gpd.GeoDataFrame):
            extent = imod.prepare.rasterize(extent, like=like)
        elif isinstance(extent, xr.DataArray):
            pass
        else:
            raise ValueError("extent must be of type tuple, GeoDataFrame or DataArray")

        extent = xr.ones_like(extent).where(extent > 0)

        # create edge around extent to apply boundary conditions
        if not (heads_boundary is None and concentration_boundary is None):
            extentwithedge = extent.copy(data=binary_dilation(extent.fillna(0).values))
            is_domain = extentwithedge == 1
        else:
            is_domain = extent == 1

        # clip to domain
        extent = extent.where(is_domain, drop=True)
        is_domain = (
            is_domain.where(is_domain, drop=True) == 1
        )  # again == 1 as it is cast to float b/c nans
        is_edge = xr.ones_like(extent).where(is_domain & extent.isnull()) == 1

        # Clip model to extent, set area outside extent to nodata or 0
        clip_slices = {"x": is_domain.x.values, "y": is_domain.y.values}
        model = self.sel(**clip_slices)
        for pkgname in model.keys():
            for variable in model[pkgname].data_vars:
                da = model[pkgname][variable]
                if variable in ["ibound", "icbund"]:
                    model[pkgname][variable] = da.where(is_domain, 0)
                # Check if geospatial (IDF) data
                elif "x" in da.dims and "y" in da.dims:
                    model[pkgname][variable] = da.where(is_domain)

        # Create boundary conditions as CHD and/or TVC package
        # Time shifts: assume heads and conc are calculation results. Then:
        # timestamp of result is _end_ of stress period, while timestamp input
        # is the _start_ of the stress period.
        if concentration_boundary is not None:
            concentration_boundary = concentration_boundary.sel(**clip_slices)
            concentration_boundary = concentration_boundary.where(is_edge)
            if "time" in concentration_boundary.dims:
                concentration_boundary = xr.concat(
                    [
                        concentration_boundary.shift(time=-1).isel(
                            time=slice(None, -1)
                        ),
                        concentration_boundary.isel(time=-1),
                    ],
                    dim="time",
                )
            model["tvc"] = imod.wq.TimeVaryingConstantConcentration(
                concentration=concentration_boundary
            )

        if heads_boundary is not None:
            heads_boundary = heads_boundary.sel(**clip_slices)
            head_end = heads_boundary.where(is_edge)
            head_start = head_end.copy()
            if "time" in head_start.dims:
                head_start = xr.concat(
                    [
                        head_start.isel(time=0),
                        head_start.shift(time=1).isel(time=slice(1, None)),
                    ],
                    dim="time",
                )

            model["chd"] = imod.wq.ConstantHead(
                head_start=head_start,
                head_end=head_end,
                concentration=concentration_boundary,  # concentration relevant for fwh calc
            )

        # delete packages if no data in sliced model
        if delete_empty_pkg:
            model._delete_empty_packages(verbose=True)

        return model
