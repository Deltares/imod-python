import abc
import pathlib

import jinja2
import numpy as np
import xarray as xr


class Package(xr.Dataset, abc.ABC):
    """
    Package is used to share methods for specific packages with no time
    component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    This class only supports `array input
    <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=16>`_,
    not the list input which is used in :class:`BoundaryCondition`.
    """

    __slots__ = ("_template", "_pkg_id", "_period_data")

    def _valid(self, value):
        """
        Filters values that are None, False, or a numpy.bool_ False.
        Needs to be this specific, since 0.0 and 0 are valid values, but are
        equal to a boolean False.
        """
        # Test singletons
        if value is False or value is None:
            return False
        # Test numpy bool (not singleton)
        elif isinstance(value, np.bool_) and not value:
            return False
        else:
            return True

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

    def to_sparse(self, arrdict, layer):
        """Convert from dense arrays to list based input"""
        # TODO add pkgcheck that period table aligns
        # Get the number of valid values
        data = next(iter(arrdict.values()))
        notnull = ~np.isnan(data)
        nrow = notnull.sum()
        # Define the numpy structured array dtype
        index_spec = [("layer", np.int32), ("row", np.int32), ("column", np.int32)]
        field_spec = [(key, np.float64) for key in arrdict]
        sparse_dtype = np.dtype(index_spec + field_spec)

        # Initialize the structured array
        recarr = np.empty(nrow, dtype=sparse_dtype)
        # Fill in the indices
        if layer is not None:
            recarr["layer"] = layer
            recarr["row"], recarr["column"] = (np.argwhere(notnull) + 1).transpose()
        else:
            recarr["layer"], recarr["row"], recarr["column"] = (
                np.argwhere(notnull) + 1
            ).transpose()

        # Fill in the data
        for key, arr in arrdict.items():
            values = arr[notnull].astype(np.float64)
            recarr[key] = values

        return recarr

    def _check_layer_presence(self, ds):
        """
        If layer present in coordinates and dimensions return layers,
        if not return None
        """

        if "layer" in ds.coords and "layer" not in ds.dims:
            layer = ds["layer"].values
        else:
            layer = None
        return layer

    def _ds_to_arrdict(self, ds):
        arrdict = {}
        for datavar in ds.data_vars:
            if ds[datavar].shape == ():
                raise ValueError(
                    f"{datavar} in {ds._pkg_id} package cannot be a scalar"
                )
            arrdict[datavar] = ds[datavar].values
        return arrdict

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
            np.savetxt(fname=f, X=da.values.reshape((1, -1)), fmt=fmt)

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        for k, v in self.data_vars.items():  # pylint:disable=no-member
            value = v.values[()]
            if self._valid(value):  # skip None and False
                d[k] = value
        return self._template.render(d)

    def _compose_values(self, da, directory, name, binary):
        """
        Compose values of dictionary.

        Ignores times. Time dependent boundary conditions use the method from
        BoundaryCondition.

        See documentation of wq
        """
        layered = False
        values = []

        if "x" in da.dims and "y" in da.dims:
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
            if "x" in self.dims and "y" in self.dims:
                pkgdirectory = directory / pkgname
                pkgdirectory.mkdir(exist_ok=True, parents=True)
                for varname, dtype in self._grid_data.items():
                    key = self._keyword_map.get(varname, varname)
                    da = self[varname]
                    if "x" in da.dims and "y" in da.dims:
                        if binary:
                            path = pkgdirectory / f"{key}.bin"
                            self.write_binary_griddata(path, da, dtype)
                        else:
                            path = pkgdirectory / f"{key}.dat"
                            self.write_text_griddata(path, da, dtype)

    def _netcdf_path(self, directory, pkgname):
        """create path for netcdf, this function is also used to create paths to use inside the qgis projectfiles"""
        return directory / pkgname / f"{self._pkg_id}.nc"

    def write_netcdf(self, directory, pkgname, aggregate_layers=False):
        """Write to netcdf. Useful for generating .qgs projectfiles to view model input.
        These files cannot be used to run a modflow model.

        Parameters
        ----------
        directory : Path
            directory of qgis project

        pkgname : str
            package name

        aggregate_layers : bool
            If True, aggregate layers by taking the mean, i.e. ds.mean(dim="layer")

        Returns
        -------
        has_dims : list of str
            list of variables that have an x and y dimension.

        """

        has_dims = []
        for varname in self.data_vars.keys():  # pylint:disable=no-member
            if all(i in self[varname].dims for i in ["x", "y"]):
                has_dims.append(varname)

        spatial_ds = self[has_dims]

        if aggregate_layers and ("layer" in spatial_ds.dims):
            spatial_ds = spatial_ds.mean(dim="layer")

        if "time" not in spatial_ds:
            # Hack to circumvent this issue:
            # https://github.com/lutraconsulting/MDAL/issues/300
            spatial_ds = spatial_ds.assign_coords(
                time=np.array("1970-01-01", dtype=np.datetime64)
            ).expand_dims(dim="time")

        path = self._netcdf_path(directory, pkgname)
        path.parent.mkdir(exist_ok=True, parents=True)

        spatial_ds.to_netcdf(path)
        return has_dims


class BoundaryCondition(Package, abc.ABC):
    """
    BoundaryCondition is used to share methods for specific stress packages with a time component.

    It is not meant to be used directly, only to inherit from, to implement new packages.

    This class only supports `list input <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=19>`_,
    not the array input which is used in :class:`Package`.
    """

    __slots__ = ()

    def _max_active_n(self):
        """
        Determine the maximum active number of cells that are active
        during a stress period.
        """
        da = self[self._period_data[0]]
        if "time" in da.coords:
            nmax = int(da.groupby("time").count(xr.ALL_DIMS).max())
        else:
            nmax = int(da.count())
        return nmax

    def _write_binaryfile(self, outpath, sparse_data, binary):
        with open(outpath, "w") as f:
            if binary:
                sparse_data.tofile(f)
            else:
                fields = sparse_data.dtype.fields
                fmt = [self._number_format(field[0]) for field in fields.values()]
                header = " ".join(list(fields.keys()))
                np.savetxt(fname=f, X=sparse_data, fmt=fmt, header=header)

    def write_datafile(self, outpath, ds, binary):
        """
        Writes a modflow6 binary data file
        """
        layer = self._check_layer_presence(ds)
        arrdict = self._ds_to_arrdict(ds)
        sparse_data = self.to_sparse(arrdict, layer)
        outpath.parent.mkdir(exist_ok=True, parents=True)
        self._write_binaryfile(outpath, sparse_data, binary)

    def period_paths(self, directory, pkgname, globaltimes, bin_ds, binary):
        if binary:
            ext = "bin"
        else:
            ext = "dat"

        periods = {}
        if "time" in bin_ds:  # one of bin_ds has time
            package_times = bin_ds.coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + 1
            for i, s in enumerate(starts):
                path = directory / pkgname / f"{self._pkg_id}-{i}.{ext}"
                periods[s] = path.as_posix()
        else:
            path = directory / pkgname / f"{self._pkg_id}.{ext}"
            periods[1] = path.as_posix()
        return periods

    def get_options(self, d, not_options=None):
        if not_options is None:
            not_options = self._period_data

        for varname in self.data_vars.keys():  # pylint:disable=no-member
            if varname in not_options:
                continue
            v = self[varname].values[()]
            if self._valid(v):  # skip None and False
                d[varname] = v
        return d

    def render(self, directory, pkgname, globaltimes, binary):
        """Render fills in the template only, doesn't write binary data"""
        d = {"binary": binary}
        bin_ds = self[list(self._period_data)]
        d["periods"] = self.period_paths(
            directory, pkgname, globaltimes, bin_ds, binary
        )
        # construct the rest (dict for render)
        d = self.get_options(d)
        d["maxbound"] = self._max_active_n()
        return self._template.render(d)

    def write_perioddata(self, directory, pkgname, binary):
        bin_ds = self[list(self._period_data)]

        if binary:
            ext = "bin"
        else:
            ext = "dat"

        if "time" in bin_ds:  # one of bin_ds has time
            for i in range(len(self.time)):
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


class AdvancedBoundaryCondition(BoundaryCondition, abc.ABC):
    """Class dedicated to advanced boundary conditions, since MF6 does not support
    binary files for Advanced Boundary conditions.

    The advanced boundary condition packages are: "uzf", "lak", "maw", "str".

    """

    __slots__ = ()

    def _get_field_spec_from_dtype(self, recarr):
        """
        From https://stackoverflow.com/questions/21777125/how-to-output-dtype-to-a-list-or-dict
        """
        return [
            (x, y[0])
            for x, y in sorted(recarr.dtype.fields.items(), key=lambda k: k[1])
        ]

    def _write_file(self, outpath, sparse_data):
        """
        Write to textfile, which is necessary for Advanced Stress Packages
        """
        fields = sparse_data.dtype.fields
        fmt = [self._number_format(field[0]) for field in fields.values()]
        header = " ".join(list(fields.keys()))
        np.savetxt(fname=outpath, X=sparse_data, fmt=fmt, header=header)

    @abc.abstractmethod
    def _package_data_to_sparse(self):
        """
        Get packagedata, override with function for the advanced boundary
        condition in particular
        """
        return

    def write_packagedata(self, directory, pkgname, binary):
        outpath = directory / pkgname / f"{self._pkg_id}-pkgdata.bin"
        outpath.parent.mkdir(exist_ok=True, parents=True)
        package_data = self._package_data_to_sparse()
        self._write_file(outpath, package_data)

    def write(self, directory, pkgname, globaltimes, binary):
        self.fill_stress_perioddata()
        self.write_blockfile(directory, pkgname, globaltimes, binary)
        self.write_perioddata(directory, pkgname, binary)
        self.write_packagedata(directory, pkgname, binary)
