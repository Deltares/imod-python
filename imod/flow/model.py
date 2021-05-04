import collections
import os
import pathlib

import cftime
import jinja2
import xarray as xr
import numpy as np
import pandas as pd

import imod

# TODO: Merge time utilities, this is becoming a mess
from imod.wq import timeutil
from imod.flow.timeutil import insert_unique_package_times
import imod.util as util
from imod.flow.pkgbase import Vividict

from imod.flow.pkggroup import PackageGroups
from imod.flow.pkgbase import BoundaryCondition

from dataclasses import dataclass

import collections
import abc


class IniFile(collections.UserDict, abc.ABC):
    """
    Some basic support for iMOD ini files here

    These files contain the settings that iMOD uses to run its' batch fucntions.
    For example to convert its' model description
    (a projectfile containing paths to respective .IDFs for each package)
    to a Modflow6 model.
    """

    # TODO: Create own key mapping to avoid keys like "edate"?
    _template = jinja2.Template(
        "{%- for key, value in settings %}\n" "{{key}}={{value}}\n" "{%- endfor %}\n"
    )

    def _format_datetimes(self):
        for timekey in ["sdate", "edate"]:
            if timekey in self.keys():
                # If not string assume it is in some kind of datetime format
                if type(self[timekey]) != str:
                    self[timekey] = util._compose_timestring(self[timekey])

    def render(self):
        self._format_datetimes()
        return self._template.render(settings=self.items())


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

    def _use_cftime(self):
        """
        Also checks if datetime types are homogeneous across packages.
        """
        types = []
        for pkg in self.values():
            if pkg._hastime():
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
        times, first_times = insert_unique_package_times(self.items(), times)

        # Check if every transient package commences at the same time.
        for key, first_time in first_times.items():
            time0 = times[0]
            if (first_time != time0) and not self[key]._is_periodic():
                raise ValueError(
                    f"Package {key} does not have a value specified for the "
                    f"first time: {time0}. Every input must be present in the "
                    "first stress period. Values are only filled forward in "
                    "time."
                )

        duration = timeutil.timestep_duration(times, self.use_cftime)
        # Generate time discretization, just rely on default arguments
        # Probably won't be used that much anyway?
        times = np.array(times)
        timestep_duration = xr.DataArray(
            duration, coords={"time": times[:-1]}, dims=("time",)
        )
        self["time_discretization"] = imod.flow.TimeDiscretization(
            timestep_duration=timestep_duration, endtime=times[-1]
        )

    def _calc_n_entry(self, composed_package, is_boundary_condition):
        """Calculate amount of entries for each timestep and variable."""

        def first(d):
            """Get first value of dictionary values"""
            return next(iter(d.values()))

        if is_boundary_condition:
            first_variable = first(first(composed_package))
            n_entry = 0
            for sys in first_variable.values():
                n_entry += len(sys)

            return n_entry

        else:  # No time and no systems in regular packages
            first_variable = first(composed_package)
            return len(first_variable)

    def _compose_timestrings(self, globaltimes):
        time_format = "%Y-%m-%d %H:%M:%S"
        time_composed = self["time_discretization"]._compose_values_time(
            "time", globaltimes
        )
        time_composed = dict(
            [
                (timestep_nr, util._compose_timestring(time, time_format=time_format))
                for timestep_nr, time in time_composed.items()
            ]
        )
        return time_composed

    def _compose_periods(self):
        periods = {}

        for key, package in self.items():
            if package._is_periodic():
                # Periodic stresses are defined for all variables
                first_var = list(package.dataset.data_vars)[0]
                periods.update(package.dataset[first_var].attrs["stress_periodic"])

        # Create timestrings for "Periods" section in projectfile
        # Basically swap around period attributes and compose timestring
        # Note that the timeformat for periods in the Projectfile is different
        # from that for stress periods
        time_format = "%d-%m-%Y %H:%M:%S"
        periods_composed = dict(
            [
                (value, util._compose_timestring(time, time_format=time_format))
                for time, value in periods.items()
            ]
        )
        return periods_composed

    def _compose_all_packages(self, directory, globaltimes, compose_projectfile=True):
        """
        Compose all transient packages before rendering.

        Required because of outer timeloop

        Returns
        -------
        A tuple with lists of respectively the composed packages and boundary conditions
        """
        bndkey = self._get_pkgkey("bnd")
        nlayer = self[bndkey]["layer"].size

        composition = Vividict()

        group_packages = self._group()

        # Get get pkg_id from first value in dictionary in group list
        group_pkg_ids = [next(iter(group.values()))._pkg_id for group in group_packages]

        for group in group_packages:
            group.compose(
                directory,
                globaltimes,
                nlayer,
                composition=composition,
                compose_projectfile=compose_projectfile,
            )

        for key, package in self.items():
            if package._pkg_id not in group_pkg_ids:
                package.compose(
                    directory.joinpath(key),
                    globaltimes,
                    nlayer,
                    composition=composition,
                    compose_projectfile=compose_projectfile,
                )

        return composition

    def _render_periods(self, periods_composed):
        _template_periods = jinja2.Template(
            "Periods\n"
            "{%- for key, timestamp in periods.items() %}\n"
            "{{key}}\n{{timestamp}}\n"
            "{%- endfor %}\n"
        )

        return _template_periods.render(periods=periods_composed)

    def _render_projectfile(self, directory):
        """
        Render projectfile. The projectfile has the hierarchy:
        package - time - system - layer
        """
        diskey = self._get_pkgkey("dis")
        globaltimes = self[diskey]["time"].values

        content = []

        composition = self._compose_all_packages(
            directory, globaltimes, compose_projectfile=True
        )

        times_composed = self._compose_timestrings(globaltimes)

        periods_composed = self._compose_periods()

        # Add period strings to times_composed
        # These are the strings atop each stress period in the projectfile
        times_composed.update({key: key for key in periods_composed.keys()})

        rendered = []
        ignored = ["dis"]

        for key, package in self.items():
            pkg_id = package._pkg_id

            if (pkg_id in rendered) or (pkg_id in ignored):
                continue  # Skip if already rendered (for groups) or not necessary to render

            kwargs = dict(
                pkg_id=pkg_id,
                name=package.__class__.__name__,
                variable_order=package._variable_order,
                package_data=composition[pkg_id],
            )

            if isinstance(package, BoundaryCondition):
                kwargs["n_entry"] = self._calc_n_entry(composition[pkg_id], True)
                kwargs["times"] = times_composed
            else:
                kwargs["n_entry"] = self._calc_n_entry(composition[pkg_id], False)

            content.append(package._render_projectfile(**kwargs))
            rendered.append(pkg_id)

        # Add periods definition
        content.append(self._render_periods(periods_composed))

        return "\n\n".join(content)

    def _render_runfile(self, directory):
        """
        Render runfile. The runfile has the hierarchy:
        time - package - system - layer
        """
        raise NotImplementedError("Currently only projectfiles can be rendered.")

    def render(self, directory, render_projectfile=True):
        """
        Render the runfile as a string, package by package.
        """
        if render_projectfile:
            return self._render_projectfile(directory)
        else:
            return self._render_runfile(directory)

    def _model_path_management(
        self, directory, result_dir, resultdir_is_workdir, render_projectfile
    ):
        # Coerce to pathlib.Path
        directory = pathlib.Path(directory)
        if result_dir is None:
            result_dir = pathlib.Path("results")
        else:
            result_dir = pathlib.Path(result_dir)

        # Create directories if necessary
        directory.mkdir(exist_ok=True, parents=True)
        result_dir.mkdir(exist_ok=True, parents=True)

        if render_projectfile:
            ext = ".prj"
        else:
            ext = ".run"

        runfilepath = directory / f"{self.modelname}{ext}"
        results_runfilepath = result_dir / f"{self.modelname}{ext}"

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

        return result_dir, render_dir, runfilepath, results_runfilepath, caching_reldir

    def write(
        self,
        directory=pathlib.Path("."),
        result_dir=None,
        resultdir_is_workdir=False,
        render_projectfile=True,
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

        # TODO: Find a cleaner way to pack and unpack these paths
        (
            result_dir,
            render_dir,
            runfilepath,
            results_runfilepath,
            caching_reldir,
        ) = self._model_path_management(
            directory, result_dir, resultdir_is_workdir, render_projectfile
        )

        directory = directory.resolve()  # Force absolute paths

        # TODO
        # Check if any caching packages are present, and set necessary states.
        # self._set_caching_packages(caching_reldir)

        if not self.check is None:
            self.package_check()

        # TODO Necessary?
        # Delete packages without data
        # self._delete_empty_packages(verbose=True)

        runfile_content = self.render(
            directory=directory, render_projectfile=render_projectfile
        )

        # Start writing
        # Write the runfile
        with open(runfilepath, "w") as f:
            f.write(runfile_content)
        # Also write the runfile in the workdir
        if resultdir_is_workdir:
            with open(results_runfilepath, "w") as f:
                f.write(runfile_content)

        # Write iMOD TIM file
        diskey = self._get_pkgkey("dis")
        time_path = directory / f"{diskey}.tim"
        self[diskey].save(time_path)

        # Create and write INI file to configure conversion/simulation
        config = IniFile(
            sim_type=2,
            function="runfile",
            prjfile_in=directory / runfilepath.name,
            namfile_out=directory / (runfilepath.stem + ".nam"),
            iss=1,
            timfname=directory / time_path.name,
        )
        config_content = config.render()

        with open(directory / "config_run.ini", "w") as f:
            f.write(config_content)

        # Write all IDFs and IPFs
        for pkgname, pkg in self.items():
            if (
                "x" in pkg.dataset.coords and "y" in pkg.dataset.coords
            ) or pkg._pkg_id in ["wel", "hfb"]:
                try:
                    pkg.save(directory=directory / pkgname)
                except Exception as error:
                    raise RuntimeError(
                        f"An error occured during saving of package: {pkgname}."
                    ) from error

    def _check_top_bottom(self):
        """Check whether bottom of a layer does not exceed a top somewhere."""
        basic_ids = ["top", "bot"]

        topkey, botkey = [self._get_pkgkey(pkg_id) for pkg_id in basic_ids]
        top, bot = [self[key] for key in (topkey, botkey)]

        if (top["top"] < bot["bottom"]).any():
            raise ValueError(
                f"top should be larger than bottom in {topkey} and {botkey}"
            )

    def package_check(self):
        bndkey = self._get_pkgkey("bnd")
        active_cells = self[bndkey]["ibound"] != 0

        self._check_top_bottom()

        for pkg in self.values():
            pkg._pkgcheck(active_cells=active_cells)
