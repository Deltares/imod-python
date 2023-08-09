"""
Module to store (prototype) low-level modflow6 package base classes.

These are closer to the Modflow 6 data model, with a cellid,
instead of x, y coordinates.

We plan to split up the present attributes of the classes in pkgbase.py
(Package and BoundaryCondition) into low-level and high-level classes.
"""

# REFACTORING STEPS
# 1. PackageBase: Copy paste relevant functions from existing Package to low-level and high-level packages.
# 2. PackageBase: Disentangle high-level and low-level data dependencies (e.g. ensure low-level class does not look for "x" and "y" keys in dimensions)
# 3. Create prototype low-level NodeProperty flow package
# 4. Create prototype high-level NodeProperty flow package
# 5. Replace existing NodeProperty flow package with high-level NodePropertyFlow package
# 6. Create prototype low-level Drain package
# 7. Create prototype high-level Drain package
# 8. Replace existing Drain package with high-level Drain package
# 9. Roll out to other packages

import abc
import pathlib

import jinja2
import numpy as np
import xarray as xr
import xugrid as xu

from imod.mf6.pkgbase_shared import NewPackageBase

TRANSPORT_PACKAGES = ("adv", "dsp", "ssm", "mst", "ist", "src")


class Mf6Pkg(NewPackageBase, abc.ABC):
    """ """

    def __init__(self, allargs=None):
        super().__init__(allargs)

    @classmethod
    def from_file(cls, path) -> "Mf6Pkg":
        """
        Instantiate class from modflow 6 file
        """
        # return cls.__new__(cls)
        raise NotImplementedError("from_file not implemented")

    @staticmethod
    def _number_format(dtype: type):
        if np.issubdtype(dtype, np.integer):
            return "%i"
        elif np.issubdtype(dtype, np.floating):
            return "%.18G"
        else:
            raise TypeError("dtype should be either integer or float")

    @staticmethod
    def _initialize_template(pkg_id):
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        if pkg_id == "ims":
            fname = "sln-ims.j2"
        elif pkg_id == "tdis":
            fname = "sim-tdis.j2"
        elif pkg_id in TRANSPORT_PACKAGES:
            fname = f"gwt-{pkg_id}.j2"
        else:
            fname = f"gwf-{pkg_id}.j2"
        return env.get_template(fname)

    def write_blockfile(self, directory, pkgname, globaltimes, binary):
        content = self.render(
            directory=directory,
            pkgname=pkgname,
            globaltimes=globaltimes,
            binary=binary,
        )
        filename = directory / f"{pkgname}.{self._pkg_id}"
        with open(filename, "w") as f:
            f.write(content)

    def write_binary_griddata(self, outpath, da, dtype):
        # From the modflow6 source, the header is defined as:
        # integer(I4B) :: kstp --> np.int32 : 1
        # integer(I4B) :: kper --> np.int32 : 2
        # real(DP) :: pertim --> 2 * np.int32 : 4
        # real(DP) :: totim --> 2 * np.int32 : 6
        # character(len=16) :: text --> 4 * np.int32 : 10
        # integer(I4B) :: m1, m2, m3 --> 3 * np.int32 : 13
        # so writing 13 bytes suffices to create a header.

        # The following code is commented out due to modflow issue 189
        # https://github.com/MODFLOW-USGS/modflow6/issues/189
        # We never write LAYERED data.
        # The (structured) dis array reader results in an error if you try to
        # read a 3D botm array. By storing nlayer * nrow * ncol in the first
        # header entry, the array is read properly.

        # haslayer = "layer" in da.dims
        # if haslayer:
        #    nlayer, nrow, ncol = da.shape
        # else:
        #    nrow, ncol = da.shape
        #    nlayer = 1

        # This is a work around for the abovementioned issue.
        nval = np.product(da.shape)
        header = np.zeros(13, np.int32)
        header[-3] = np.int32(nval)  # ncol
        header[-2] = np.int32(1)  # nrow
        header[-1] = np.int32(1)  # nlayer

        with open(outpath, "w") as f:
            header.tofile(f)
            da.values.flatten().astype(dtype).tofile(f)

    def write_text_griddata(self, outpath, da, dtype):
        with open(outpath, "w") as f:
            # Note: reshaping here avoids writing newlines after every number.
            # This dumps all the values in a single row rather than a single
            # column. This is to be preferred, since editors can easily
            # "reshape" a long row with "word wrap"; they cannot as easily
            # ignore newlines.
            fmt = self._number_format(dtype)
            data = da.values
            if data.ndim > 2:
                np.savetxt(fname=f, X=da.values.reshape((1, -1)), fmt=fmt)
            else:
                np.savetxt(fname=f, X=da.values, fmt=fmt)

    # TODO: Change variable order, so that globatimes can be changed to *args
    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        if directory is None:
            pkg_directory = pkgname
        else:
            pkg_directory = pathlib.Path(directory.stem) / pkgname

        for varname in self.dataset.data_vars:
            key = self._keyword_map.get(varname, varname)

            if hasattr(self, "_grid_data") and varname in self._grid_data:
                layered, value = self._compose_values(
                    self.dataset[varname], pkg_directory, key, binary=binary
                )
                if self._valid(value):  # skip False or None
                    d[f"{key}_layered"], d[key] = layered, value
            else:
                value = self[varname].values[()]
                if self._valid(value):  # skip False or None
                    d[key] = value
        return self._template.render(d)

    # TODO: Find alternative for this, "x" and "y" dimensions
    # will not be in this class' dataset anymore. For example, check cellid present.
    @staticmethod
    def _is_xy_data(obj):
        if isinstance(obj, (xr.DataArray, xr.Dataset)):
            xy = "x" in obj.dims and "y" in obj.dims
        elif isinstance(obj, (xu.UgridDataArray, xu.UgridDataset)):
            xy = obj.ugrid.grid.face_dimension in obj.dims
        else:
            raise TypeError(
                "obj should be DataArray or UgridDataArray, "
                f"received {type(obj)} instead"
            )
        return xy

    def _compose_values(self, da, directory, name, binary):
        """
        Compose values of dictionary.

        Ignores times. Time dependent boundary conditions use the method from
        BoundaryCondition.

        See documentation of wq
        """
        layered = False
        values = []
        if self._is_xy_data(da):
            if binary:
                path = (directory / f"{name}.bin").as_posix()
                values.append(f"open/close {path} (binary)")
            else:
                path = (directory / f"{name}.dat").as_posix()
                values.append(f"open/close {path}")
        else:
            if "layer" in da.dims:
                layered = True
                for layer in da.coords["layer"]:
                    values.append(f"constant {da.sel(layer=layer).values[()]}")
            else:
                value = da.values[()]
                if self._valid(value):  # skip None or False
                    values.append(f"constant {value}")
                else:
                    values = None

        return layered, values

    def write(self, directory, pkgname, globaltimes, binary):
        directory = pathlib.Path(directory)
        self.write_blockfile(directory, pkgname, globaltimes, binary=binary)

        if hasattr(self, "_grid_data"):
            if self._is_xy_data(self.dataset):
                pkgdirectory = directory / pkgname
                pkgdirectory.mkdir(exist_ok=True, parents=True)
                for varname, dtype in self._grid_data.items():
                    key = self._keyword_map.get(varname, varname)
                    da = self.dataset[varname]
                    if self._is_xy_data(da):
                        if binary:
                            path = pkgdirectory / f"{key}.bin"
                            self.write_binary_griddata(path, da, dtype)
                        else:
                            path = pkgdirectory / f"{key}.dat"
                            self.write_text_griddata(path, da, dtype)

    def get_auxiliary_variable_names(self):
        result = {}
        if hasattr(self, "_auxiliary_data"):
            result.update(self._auxiliary_data)
        return result


class Mf6TimeVaryingPkg(Mf6Pkg):
    """
    Low-level package to share methods for specific modflow 6 packages with
    time component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    This class only supports `list input
    <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=19>`_,
    not the array input which is used in :class:`Mf6Pkg`.
    """

    # TODO: Rename to `to_struct_array`
    # NOTE: This function should be part of Mf6Pkg, now defined here,
    #   to ensure function is overrided.
    def to_sparse(self, ds):
        data_vars = [var for var in ds.data_vars if var != "cellid"]
        cellid_names = ds.coords["nmax_cellid"].values
        nrow = ds.coords["ncellid"].size

        index_spec = [(index, np.int32) for index in cellid_names]
        field_spec = [(var, np.float64) for var in data_vars]
        sparse_dtype = np.dtype(index_spec + field_spec)

        # Initialize the structured array
        recarr = np.empty(nrow, dtype=sparse_dtype)
        for cellid_name in cellid_names:
            recarr[cellid_name] = ds["cellid"].sel(nmax_cellid=cellid_name).values
        for var in data_vars:
            recarr[var] = ds[var]
        return recarr

    def write_datafile(self, outpath, ds, binary):
        """
        Writes a modflow6 binary data file
        """
        sparse_data = self.to_sparse(ds)
        outpath.parent.mkdir(exist_ok=True, parents=True)
        if binary:
            self._write_binaryfile(outpath, sparse_data)
        else:
            self._write_textfile(outpath, sparse_data)

    def _write_binaryfile(self, outpath, sparse_data):
        with open(outpath, "w") as f:
            sparse_data.tofile(f)

    def _write_textfile(self, outpath, sparse_data):
        fields = sparse_data.dtype.fields
        fmt = [self._number_format(field[0]) for field in fields.values()]
        header = " ".join(list(fields.keys()))
        with open(outpath, "w") as f:
            np.savetxt(fname=f, X=sparse_data, fmt=fmt, header=header)

    def _max_active_n(self):
        """
        Determine the maximum active number of cells that are active
        during a stress period.
        """
        da = self.dataset[self.period_data()[0]]
        if "time" in da.coords:
            nmax = int(da.groupby("time").count(xr.ALL_DIMS).max())
        else:
            nmax = int(da.count())
        return nmax

    def period_paths(self, directory, pkgname, globaltimes, bin_ds, binary):
        pkg_directory = pathlib.Path(directory.stem) / pkgname

        if binary:
            ext = "bin"
        else:
            ext = "dat"

        periods = {}
        if "time" in bin_ds:  # one of bin_ds has time
            package_times = bin_ds.coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + 1
            for i, start in enumerate(starts):
                path = pkg_directory / f"{self._pkg_id}-{i}.{ext}"
                periods[start] = path.as_posix()

            repeat_stress = self.dataset.get("repeat_stress")
            if repeat_stress is not None and repeat_stress.values[()] is not None:
                keys = repeat_stress.isel(repeat_items=0).values
                values = repeat_stress.isel(repeat_items=1).values
                repeat_starts = np.searchsorted(globaltimes, keys) + 1
                values_index = np.searchsorted(globaltimes, values) + 1
                for i, start in zip(values_index, repeat_starts):
                    periods[start] = periods[i]
                # Now make sure the periods are sorted by key.
                periods = dict(sorted(periods.items()))
        else:
            path = pkg_directory / f"{self._pkg_id}.{ext}"
            periods[1] = path.as_posix()

        return periods

    def get_options(self, d, not_options=None):
        if not_options is None:
            not_options = self.period_data()

        for varname in self.dataset.data_vars.keys():  # pylint:disable=no-member
            if varname in not_options:
                continue
            v = self.dataset[varname].values[()]
            if self._valid(v):  # skip None and False
                d[varname] = v
        return d

    def render(self, directory, pkgname, globaltimes, binary):
        """Render fills in the template only, doesn't write binary data"""
        d = {"binary": binary}
        bin_ds = self[self.period_data()]
        d["periods"] = self.period_paths(
            directory, pkgname, globaltimes, bin_ds, binary
        )
        # construct the rest (dict for render)
        d = self.get_options(d)
        d["maxbound"] = self._max_active_n()

        # now we should add the auxiliary variable names to d
        auxiliaries = (
            self.get_auxiliary_variable_names()
        )  # returns someting like {"concentration": "species"}

        # loop over the types of auxiliary variables (for example concentration)
        for auxvar in auxiliaries.keys():
            # if "concentration" is a variable of this dataset
            if auxvar in self.dataset.data_vars:
                # if our concentration dataset has the species coordinate
                if auxiliaries[auxvar] in self.dataset[auxvar].coords:
                    # assign the species names list to d
                    d["auxiliary"] = self.dataset[auxiliaries[auxvar]].values
                else:
                    # the error message is more specific than the code at this point.
                    raise ValueError(
                        f"{auxvar} requires a {auxiliaries[auxvar]} coordinate."
                    )

        return self._template.render(d)

    def write_perioddata(self, directory, pkgname, binary):
        if len(self.period_data()) == 0:
            return
        bin_ds = self[self.period_data()]

        if binary:
            ext = "bin"
        else:
            ext = "dat"

        if "time" in bin_ds:  # one of bin_ds has time
            for i in range(len(self.dataset.time)):
                path = directory / pkgname / f"{self._pkg_id}-{i}.{ext}"
                self.write_datafile(
                    path, bin_ds.isel(time=i), binary=binary
                )  # one timestep
        else:
            path = directory / pkgname / f"{self._pkg_id}.{ext}"
            self.write_datafile(path, bin_ds, binary=binary)

    def write(self, directory, pkgname, globaltimes, binary):
        """
        writes the blockfile and binary data

        directory is modelname
        """
        directory = pathlib.Path(directory)
        self.write_blockfile(
            directory=directory,
            pkgname=pkgname,
            globaltimes=globaltimes,
            binary=binary,
        )
        self.write_perioddata(
            directory=directory,
            pkgname=pkgname,
            binary=binary,
        )
