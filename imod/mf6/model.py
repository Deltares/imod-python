import collections
import os
import pathlib

import cftime
import numpy as np
import xarray as xr

import imod


class Model(collections.UserDict):
    def __setitem__(self, key, value):
        # TODO: Add packagecheck
        super(__class__, self).__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class Modflow6(Model):
    """
    Contains data and writes consistent model input files
    """

    def __init__(self, modelname):
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

    def _use_cftime(self):
        """
        Also checks if datetime types are homogeneous across packages.
        """
        types = [
            type(pkg["time"].values[0]) for pkg in self.values() if "time" in pkg.coords
        ]
        # Types will be empty if there's no time dependent input
        if len(set(types)) == 0:
            return False
        else:  # there is time dependent input
            if not len(set(types)) == 1:
                raise ValueError(
                    "Multiple datetime types detected. "
                    "Use either cftime or numpy.datetime64[ns]."
                )
            # Since we compare types and not instances, we use issubclass
            if issubclass(types[0], cftime.datetime):
                return True
            elif issubclass(types[0], np.datetime64):
                return False
            else:
                raise ValueError("Use either cftime or numpy.datetime64[ns].")

    def time_discretization(self, endtime, starttime=None, *times):
        """
        Collect all unique times
        """
        self.use_cftime = self._use_cftime()

        times = [imod.wq.timeutil.to_datetime(time, self.use_cftime) for time in times]
        for pkg in self.values():
            if "time" in pkg.coords:
                times.append(pkg["time"].values)

        # TODO: check that endtime is later than all other times.
        times.append(imod.wq.timeutil.to_datetime(endtime, self.use_cftime))
        if starttime is not None:
            times.append(imod.wq.timeutil.to_datetime(starttime, self.use_cftime))

        # np.unique also sorts
        times = np.unique(np.hstack(times))

        duration = imod.wq.timeutil.timestep_duration(times, self.use_cftime)
        # Generate time discretization, just rely on default arguments
        # Probably won't be used that much anyway?
        timestep_duration = xr.DataArray(
            duration, coords={"time": np.array(times)[:-1]}, dims=("time",)
        )
        self["time_discretization"] = imod.wq.TimeDiscretization(
            timestep_duration=timestep_duration
        )

    def write(self, directory=".", result_dir=None):
        if isinstance(directory, str):
            directory = pathlib.Path(directory).joinpath(self.modelname)
        if result_dir is None:
            result_dir = "results"
        else:
            result_dir = pathlib.Path(os.path.relpath(result_dir, directory))

        # # Start writing
        # directory.mkdir(exist_ok=True, parents=True)

        # # Write the runfile
        # with open(runfilepath, "w") as f:
        #     f.write(runfile_content)

        # # Write all IDFs and IPFs
        # for pkgname, pkg in self.items():
        #     if "x" in pkg.coords and "y" in pkg.coords or pkg._pkg_id == "wel":
        #         pkg.save(directory=directory.joinpath(pkgname))
