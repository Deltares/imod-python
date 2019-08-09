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

    _pkg_id = "model"

    def __init__(self, newton=False, under_relaxation=False):
        self.newton = newton
        self.under_relaxation = under_relaxation

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

    def _yield_times(self):
        modeltimes = []
        for pkg in self.values():
            if "time" in pkg.coords:
                modeltimes.append(pkg["times"].values)
        return modeltimes

    def render(self):
        """Render model namefile"""
        d = {"newton": self["newton"], "under_relaxation": self["under_relaxation"]}
        packages = {}
        for pkgname, pkg in self.items():
            # Add the six to the package id
            pkg_id = pkg._pkg_id
            key = f"{pkg_id}6"
            packages[key] = f"{pkgname}.{pkg_id}"
        d["packages"] = packages
        return self._template.render(**d)

    def write(self, modeldirectory, globaltimes):
        """
        Write model namefile
        Write packages
        """
        modeldirectory.mkdir(exist_ok=True, parents=True)

        # write model namefile
        namefile_content = self.render()
        namefile_path = modeldirectory / f"{modelname}.nam"
        with open(namefile_path, "w") as f:
            f.write(namefile_content)

        # write package contents
        for pkgname, pkg in self.items():
            pkg.write(modeldirectory, pkgname, globaltimes)
