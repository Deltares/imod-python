import collections
import os
import pathlib
import subprocess

import cftime
import jinja2
import xarray as xr
import numpy as np
import pandas as pd

import imod
from imod.wq import timeutil
from imod.flow.pkggroup import PackageGroups
from imod.flow.pkgbase import BoundaryCondition

from dataclasses import dataclass

@dataclass
class ComposedRecord:
    "Dataclass for keeping a record of composed data"
    pkg_id: str
    key: str
    variable: str
    variable_data: dict

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

    def _delete_empty_packages(self, verbose=False):
        to_del = []
        for pkg in self.keys():
            dv = list(self[pkg].dataset.data_vars)[0]
            if not self[pkg][dv].notnull().any().compute():
                if verbose:
                    print(f"Deleting package {pkg}, found no data in parameter {dv}")
                to_del.append(pkg)
        for pkg in to_del:
            del self[pkg]


class ImodflowModel(Model):
    """
    Attributes
    ----------
    modelname : str
    check : str, optional
        When to perform model checks {None, "defer", "eager"}.
        Defaults to "defer".

    Examples
    --------

    >>> m = Imodflow("example")
    >>> m["riv"] = River(...)
    >>> # ...etc.
    >>> m.time_discretization(endtime)
    >>> m.write()
    """

    # These templates end up here since they require global information
    # from more than one package
    _PACKAGE_GROUPS = PackageGroups

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
        return (pkg._pkg_id == "wel" and "time" in pkg) or ("time" in pkg.dataset.coords)

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
        Collect all unique times from model packages and additional given `times`. These
        unique times are used as stress periods in the model. All stress packages must
        have the same starting time.

        The time discretization in imod-python works as follows:

        - The datetimes of all packages you send in are always respected
        - Subsequently, the input data you use is always included fully as well
        - All times are treated as starting times for the stress: a stress is always applied until the next specified date
        - For this reason, a final time is required to determine the length of the last stress period
        - Additional times can be provided to force shorter stress periods & more detailed output
        - Every stress has to be defined on the first stress period (this is a modflow requirement)

        Or visually (every letter a date in the time axes)::

            recharge a - b - c - d - e - f
            river    g - - - - h - - - - j
            times    - - - - - - - - - - - i

            model    a - b - c h d - e - f i

        with the stress periods defined between these dates. I.e. the model times are the set of all times you include in the model.

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
                for var in pkg.dataset.data_vars:
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
        self["time_discretization"] = imod.flow.TimeDiscretization(
            timestep_duration=timestep_duration
        )
    
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
        return content

    def _compose_timesstrings(self, globaltimes):
        time_composed = self["time_discretization"]._compose_values_time("time", globaltimes)
        time_composed = dict(
            [
                (timestep_nr, util._compose_timesstring(time)) 
                for timestep_nr, time in time_composed.items
            ]
        )
        return time_composed

    def _precompose_all_packages(self, directory, globaltimes, nlay):
        """compose all transient packages before rendering. 
        
        Required because of outer timeloop

        Returns
        -------
        A tuple with lists of respectively the composed packages and boundary conditions

        """       
        composed_packages = []
        composed_boundary_conditions = []

        for key, package in self.items():
            if isinstance(package, BoundaryCondition):
                for variable in package.dataset.data_vars:
                    composed_data = package._compose_values_timelayer(variable, globaltimes, directory, nlay)
                    record = ComposedRecord(
                        pkg_id = package._pkg_id, key = key, 
                        variable = variable, variable_data = composed_data
                        )
                    composed_boundary_conditions.append(record)
            
            elif package._pkg_id not in ["dis", "oc", "pcg"]:
                for variable in package.dataset.data_vars:
                    composed_data = package._compose_values_layer(variable, directory, nlay)
                    record = ComposedRecord(
                        pkg_id = package._pkg_id, key = key, 
                        variable = variable, variable_data = composed_data
                        )
                    composed_packages.append(record)

        return composed_packages, composed_boundary_conditions

    def _render_projectfile(self, directory, globaltimes, composed_data):
        """Render runfile. The runfile has the hierarchy:
        package - time - system - layer
        """

        _package_header_template = "\n{ntimesteps:03d}, ({pkg_id}), 1, {variable_order}"

        composed_pkgs, composed_bcs = composed_data
        sorted(composed_bcs, key=lambda x: (x.pkg_id, x.key, x.variable), reverse=True)

        content = []

        # Render static packages
        for record in composed_pkgs:
            package = self[record.key]
            header = _package_header_template.format(ntimesteps = 1, pkg_id = record.pkg_id, variable_order = package.variable_order
            content.append(header)
            content.append(package.render(render_projectfile=True, **record.__dict__))

        #Render boundary conditions
        pacakge_group_dict = {}
        package_groups = self._group()

        for group in package_groups:
            pkg_id = next(iter(group.values()))._pkg_id
            package_group_dict[pkg_id] = group

        #Perhaps this data should not have been flattened
        i_start = 0
        while i <= len(composed_bcs):
            pkg_id = composed_bcs[i_start].pkg_id
            n_timesteps = len(composed_bcs[i_start].variable_data)

            if pkg_id in package_groups.keys():
                group = package_group_dict[pkg_id]
                n_var = group._n_variables
                n_sys = len(group)
            else:
                n_var = len(self[pkg_id].data_vars)
                n_sys = 1

            n_sub = n_var * n_sys
            i_end = i_start + n_sub

            for i in range(i_start, i_end):
                record = composed_bcs[i]
                package = self[record.key]
                for time, var_data in record.variable_data.items():
                    # copy dict
                    d = dict(k, v for k, v in record.__dict__.items())
                    # 
                    d["variable_data"] = var_data 

                    content.append(package.render())


            content.append()

            i_start = i_end


        for time in globaltimes:
            n_var = 
            content.append(self._render_runfile_timestep(directory, time, composed_data))
        # multi-system package group: chd, drn, ghb, riv, wel

    def _render_runfile(self, directory, globaltimes, composed_data):
        pass

    def render(self, directory, result_dir, render_projectfile=True):
        """
        Render the runfile as a string, package by package.
        """
        diskey = self._get_pkgkey("dis")
        globaltimes = self[diskey]["time"].values
        baskey = self._get_pkgkey("bas6")
        nlayer = self[baskey]["layer"].size

        composed_data = _precompose_all_packages(self, globaltimes, nlay)
        
        if render_projectfile:
            self._render_projectfile(directory, globaltimes, composed_data)
        else:
            self._render_runfile(directory, globaltimes, composed_data)

        return "\n\n".join(content)

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
            if "x" in pkg.dataset.coords and "y" in pkg.dateset.coords or pkg._pkg_id == "wel":
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