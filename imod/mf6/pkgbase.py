import abc
import dataclasses
import operator
import pathlib
from dataclasses import dataclass
from typing import Optional, Union

import jinja2
import numpy as np
import xarray as xr
import xugrid as xu


def dis_recarr(arrdict, layer, notnull):
    # Define the numpy structured array dtype
    index_spec = [("layer", np.int32), ("row", np.int32), ("column", np.int32)]
    field_spec = [(key, np.float64) for key in arrdict]
    sparse_dtype = np.dtype(index_spec + field_spec)
    # Initialize the structured array
    nrow = notnull.sum()
    recarr = np.empty(nrow, dtype=sparse_dtype)
    # Fill in the indices
    if notnull.ndim == 2:
        recarr["row"], recarr["column"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, irow, icolumn = np.argwhere(notnull).transpose()
        recarr["row"] = irow + 1
        recarr["column"] = icolumn + 1
        recarr["layer"] = layer[ilayer]
    return recarr


def disv_recarr(arrdict, layer, notnull):
    # Define the numpy structured array dtype
    index_spec = [("layer", np.int32), ("cell2d", np.int32)]
    field_spec = [(key, np.float64) for key in arrdict]
    sparse_dtype = np.dtype(index_spec + field_spec)
    # Initialize the structured array
    nrow = notnull.sum()
    recarr = np.empty(nrow, dtype=sparse_dtype)
    # Fill in the indices
    if layer.size == 1:
        recarr["cell2d"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, icell2d = np.argwhere(notnull).transpose()
        recarr["cell2d"] = icell2d + 1
        recarr["layer"] = layer[ilayer]
    return recarr


@dataclass
class VariableMetaData:
    """
    Dataclass to store metadata of a variable.

    Currently purely used to store datatypes and value limits, and can be later
    expanded to store keyword maps, and period/package data flags.
    """

    dtype: type
    not_less_than: Optional[Union[int, float]] = None
    not_less_equal_than: Optional[Union[int, float]] = None
    not_greater_than: Optional[Union[int, float]] = None
    not_greater_equal_than: Optional[Union[int, float]] = None


class Package(abc.ABC):
    """
    Package is used to share methods for specific packages with no time
    component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    This class only supports `array input
    <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=16>`_,
    not the list input which is used in :class:`BoundaryCondition`.
    """

    @classmethod
    def from_file(cls, path, **kwargs):
        """
        Loads an imod mf6 package from a file (currently only netcdf and zarr are supported).
        Note that it is expected that this file was saved with imod.mf6.Package.dataset.to_netcdf(),
        as the checks upon package initialization are not done again!

        Parameters
        ----------
        path : str, pathlib.Path
            Path to the file.
        **kwargs : keyword arguments
            Arbitrary keyword arguments forwarded to ``xarray.open_dataset()``, or
            ``xarray.open_zarr()``.
        Refer to the examples.

        Returns
        -------
        package : imod.mf6.Package
            Returns a package with data loaded from file.

        Examples
        --------

        To load a package from a file, e.g. a River package:

        >>> river = imod.mf6.River.from_file("river.nc")

        For large datasets, you likely want to process it in chunks. You can
        forward keyword arguments to ``xarray.open_dataset()`` or
        ``xarray.open_zarr()``:

        >>> river = imod.mf6.River.from_file("river.nc", chunks={"time": 1})

        Refer to the xarray documentation for the possible keyword arguments.
        """

        # Throw error if user tries to use old functionality
        if "cache" in kwargs:
            if kwargs["cache"] is not None:
                raise NotImplementedError(
                    "Caching functionality in pkg.from_file() is removed."
                )

        path = pathlib.Path(path)

        # See https://stackoverflow.com/a/2169191
        # We expect the data in the netcdf has been saved a a package
        # thus the checks run by __init__ and __setitem__ do not have
        # to be called again.
        return_cls = cls.__new__(cls)

        if path.suffix in (".zip", ".zarr"):
            # TODO: seems like a bug? Remove str() call if fixed in xarray/zarr
            return_cls.dataset = xr.open_zarr(str(path), **kwargs)
        else:
            return_cls.dataset = xr.open_dataset(path, **kwargs)

        return return_cls

    def __init__(self, allargs=None):
        if allargs is not None:
            for arg in allargs.values():
                if isinstance(arg, xu.UgridDataArray):
                    self.dataset = xu.UgridDataset(grid=arg.ugrid.grid)
                    return
        self.dataset = xr.Dataset()

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def __setitem__(self, key, value):
        self.dataset.__setitem__(key, value)

    def isel(self):
        raise NotImplementedError(
            "Selection on packages not yet supported. To make a selection on "
            f"the xr.Dataset, call {self._pkg_id}.dataset.isel instead."
            "You can create a new package with a selection by calling "
            f"{__class__.__name__}(**{self._pkg_id}.dataset.isel(**selection))"
        )

    def sel(self):
        raise NotImplementedError(
            "Selection on packages not yet supported. To make a selection on "
            f"the xr.Dataset, call {self._pkg_id}.dataset.sel instead. "
            "You can create a new package with a selection by calling "
            f"{__class__.__name__}(**{self._pkg_id}.dataset.sel(**selection))"
        )

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
        renderdir = pathlib.Path(directory.stem)
        content = self.render(
            directory=renderdir,
            pkgname=pkgname,
            globaltimes=globaltimes,
            binary=binary,
        )
        filename = directory / f"{pkgname}.{self._pkg_id}"
        with open(filename, "w") as f:
            f.write(content)

    def to_sparse(self, arrdict, layer):
        """Convert from dense arrays to list based input"""
        # TODO stream the data per stress period
        # TODO add pkgcheck that period table aligns
        # Get the number of valid values
        data = next(iter(arrdict.values()))
        notnull = ~np.isnan(data)

        if isinstance(self.dataset, xr.Dataset):
            recarr = dis_recarr(arrdict, layer, notnull)
        elif isinstance(self.dataset, xu.UgridDataset):
            recarr = disv_recarr(arrdict, layer, notnull)
        else:
            raise TypeError(
                "self.dataset should be xarray.Dataset or xugrid.UgridDataset,"
                f" is {type(self.dataset)} instead"
            )
        # Fill in the data
        for key, arr in arrdict.items():
            values = arr[notnull].astype(np.float64)
            recarr[key] = values

        return recarr

    def _ds_to_arrdict(self, ds):
        arrdict = {}
        for datavar in ds.data_vars:
            if ds[datavar].shape == ():
                raise ValueError(
                    f"{datavar} in {self._pkg_id} package cannot be a scalar"
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
        for k, v in self.dataset.data_vars.items():  # pylint:disable=no-member
            value = v.values[()]
            if self._valid(value):  # skip None and False
                d[k] = value
        return self._template.render(d)

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
        for varname in self.dataset.data_vars.keys():  # pylint:disable=no-member
            if all(i in self.dataset[varname].dims for i in ["x", "y"]):
                has_dims.append(varname)

        spatial_ds = self.dataset[has_dims]

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

    def _get_vars_to_check(self):
        """
        Helper function to get all variables which were not set to None
        """
        variables = []
        for var in self._metadata_dict.keys():
            if (  # Filter optional variables not filled in
                self.dataset[var].size != 1
            ) or (
                self.dataset[var] != None  # noqa: E711
            ):
                variables.append(var)

        return variables

    def _check_range(self):
        """
        Check if variables are within the appropriate range.
        First check all variables construct an error message with all the erronous , raise error.
        """
        variables = self._get_vars_to_check()
        pkgname = self.__class__.__name__

        range_operators = {
            "not_less_than": (operator.lt, "less than"),
            "not_less_equal_than": (operator.le, "less or equal than"),
            "not_greater_than": (operator.gt, "greater than"),
            "not_greater_equal_than": (operator.ge, "greater or equal than"),
        }

        raise_error = False
        msg = f"Detected incorrect values in {pkgname}: \n"

        for varname in variables:
            var_metadata_dict = dataclasses.asdict(self._metadata_dict[varname])

            for key, (operator_func, msg_operator) in range_operators.items():
                bound = var_metadata_dict[key]

                if (bound is not None) and (
                    operator_func(self.dataset[varname], bound).any()
                ):
                    raise_error = True
                    msg += f"- {varname} in {pkgname}: values {msg_operator} {bound} detected. \n"

        if raise_error:
            raise ValueError(msg)

    def _check_types(self):
        """Check that data types of grid data are correct."""

        variables = self._get_vars_to_check()

        for varname in variables:
            expected_dtype = self._metadata_dict[varname].dtype
            da = self.dataset[varname]

            if not issubclass(da.dtype.type, expected_dtype):
                raise TypeError(
                    f"Unexpected data type for {varname} "
                    f"in {self.__class__.__name__} package. "
                    f"Expected subclass of {expected_dtype.__name__}, "
                    f"instead got {da.dtype.type.__name__}."
                )

    def _unstructured_grid_dim_check(self, da):
        """
        Check dimension integrity of unstructured grid,
        no time dimension is accepted, data is assumed static.
        """
        if da.ndim == 0:
            return  # Scalar, no check necessary
        elif da.ndim == 1:
            face_dim = da.ugrid.grid.face_dimension
            if (face_dim not in da.dims) and ("layer" not in da.dims):
                raise ValueError(
                    f"Face dimension '{face_dim}' or dimension 'layer' "
                    f"not found in 1D UgridDataArray. "
                    f"Instead got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )
        elif da.ndim == 2:
            face_dim = da.ugrid.grid.face_dimension
            if da.dims != ("layer", face_dim):
                raise ValueError(
                    f"2D grid should have dimensions ('layer', {face_dim})"
                    f"Instead got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )

    def _structured_grid_dim_check(self, da):
        """
        Check dimension integrity of structured grid,
        no time dimension is accepted, data is assumed static.
        """
        if da.ndim == 0:
            return  # Scalar, no check necessary
        elif da.ndim == 1:
            if "layer" not in da.dims:
                raise ValueError(
                    f"1D DataArray dims can only be ('layer',). "
                    f"Instead got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )
        elif da.ndim == 2:
            if da.dims != ("y", "x"):
                raise ValueError(
                    f"2D grid should have dimensions ('y', 'x'). "
                    f"Instead, got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )
        elif da.ndim == 3:
            if da.dims != ("layer", "y", "x"):
                raise ValueError(
                    f"3D grid should have dimensions ('layer', 'y', 'x'). "
                    f"Instead, got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )
        else:
            raise ValueError(
                f"Exceeded accepted amount of dimensions for "
                f"for {da.name} in the "
                f"{self.__class__.__name__} package. "
                f"Got {da.dims}. Can be at max ('layer', 'y', 'x')."
            )

    def _check_dim_integrity(self):

        variables = self._get_vars_to_check()

        for var in variables:
            da = self.dataset[var]
            if isinstance(da, (xr.DataArray, xr.Dataset)):
                self._structured_grid_dim_check(da)
            elif isinstance(da, (xu.UgridDataArray, xu.UgridDataset)):
                self._unstructured_grid_dim_check(da)

    def _check_dim_monotonicity(self):
        """
        Check that dimensions are all monotonically increasing, or decreasing
        in the case of ``y``.
        """

        variables = self._get_vars_to_check()
        # If no variables to check return
        if len(variables) == 0:
            return

        ds = self.dataset[variables]

        for dim in ["x", "layer", "time"]:
            if dim in ds.indexes:
                if not ds.indexes[dim].is_monotonic_increasing:
                    raise ValueError(
                        f"{dim} coordinate in {self.__class__.__name__} not monotonically increasing"
                    )

        if "y" in ds.indexes:
            if not ds.indexes["y"].is_monotonic_decreasing:
                raise ValueError(
                    f"y coordinate in {self.__class__.__name__} not monotonically decreasing"
                )

    def _pkgcheck(self):
        self._check_types()
        self._check_range()
        self._check_dim_monotonicity()
        self._check_dim_integrity()


class BoundaryCondition(Package, abc.ABC):
    """
    BoundaryCondition is used to share methods for specific stress packages with a time component.

    It is not meant to be used directly, only to inherit from, to implement new packages.

    This class only supports `list input <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=19>`_,
    not the array input which is used in :class:`Package`.
    """

    def _max_active_n(self):
        """
        Determine the maximum active number of cells that are active
        during a stress period.
        """
        da = self.dataset[self._period_data[0]]
        if "time" in da.coords:
            nmax = int(da.groupby("time").count(xr.ALL_DIMS).max())
        else:
            nmax = int(da.count())
        return nmax

    def _write_binaryfile(self, outpath, sparse_data):
        with open(outpath, "w") as f:
            sparse_data.tofile(f)

    def _write_textfile(self, outpath, sparse_data):
        fields = sparse_data.dtype.fields
        fmt = [self._number_format(field[0]) for field in fields.values()]
        header = " ".join(list(fields.keys()))
        with open(outpath, "w") as f:
            np.savetxt(fname=f, X=sparse_data, fmt=fmt, header=header)

    def write_datafile(self, outpath, ds, binary):
        """
        Writes a modflow6 binary data file
        """
        layer = ds["layer"].values
        arrdict = self._ds_to_arrdict(ds)
        sparse_data = self.to_sparse(arrdict, layer)
        outpath.parent.mkdir(exist_ok=True, parents=True)
        if binary:
            self._write_binaryfile(outpath, sparse_data)
        else:
            self._write_textfile(outpath, sparse_data)

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

    def _check_all_nan(self):
        """
        Check if not grids with only nans are provided, as MAXBOUND cannot be 0.

        ``self._max_active_n()`` only determines maxbound based on the first
        variable in period_data. However, ``self._check_nan_consistent()``
        checks if nans are inconsistent amongst variables, so this catches the
        case where an all nan grid is provided for only one variable.
        """

        maxbound = self._max_active_n()
        if maxbound == 0:
            raise ValueError(
                f"Provided grid with only nans in {self.__class__.__name__}."
            )

    def _unstructured_grid_dim_check(self, da):
        """
        Check dimension integrity of unstructured grid,
        no time dimension is accepted, data is assumed static.
        """
        if da.ndim < 1:
            raise ValueError(
                f"Boundary conditions should be specified as spatial grids. "
                f"Instead, got {da.dims} for {da.name} in the "
                f"{self.__class__.__name__} package. "
            )
        elif da.ndim == 1:
            face_dim = da.ugrid.grid.face_dimension
            if face_dim not in da.dims:
                raise ValueError(
                    f"Face dimension '{face_dim}' not found in "
                    f"1D UgridDataArray. "
                    f"Instead got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )
        elif da.ndim == 2:
            face_dim = da.ugrid.grid.face_dimension
            if (da.dims != ("layer", face_dim)) and (da.dims != ("time", face_dim)):
                raise ValueError(
                    f"2D grid should have dimensions ('layer', {face_dim}) "
                    f"or ('time', {face_dim}). "
                    f"Instead got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )
        elif da.ndim == 3:
            face_dim = da.ugrid.grid.face_dimension
            if da.dims != ("time", "layer", {face_dim}):
                raise ValueError(
                    f"3D grid should have dimensions ('time', 'layer', {face_dim}) "
                    f"Instead got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )

    def _structured_grid_dim_check(self, da):
        """
        Check dimension integrity of structured grid
        """
        if da.ndim < 2:
            raise ValueError(
                f"Boundary conditions should be specified as spatial grids. "
                f"Instead, got {da.dims} for {da.name} in the "
                f"{self.__class__.__name__} package. "
            )
        elif da.ndim == 2:
            if da.dims != ("y", "x"):
                raise ValueError(
                    f"2D grid should have dimensions ('y', 'x'). "
                    f"Instead, got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )
            if "layer" not in da.coords:
                raise ValueError(
                    f"No 'layer' coordinate assigned to {da.name} "
                    f"in the {self.__class__.__name__} package. "
                    f"2D grids still require a 'layer' coordinate. "
                    f"You can assign one with `da.assign_coordinate(layer=1)`"
                )
        elif da.ndim == 3:
            if (da.dims != ("layer", "y", "x")) and (da.dims != ("time", "y", "x")):
                raise ValueError(
                    f"3D grid should have dimensions ('layer', 'y', 'x') "
                    f"or ('time', 'y', 'x'). "
                    f"Instead, got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )
        elif da.ndim == 4:
            if da.dims != ("time", "layer", "y", "x"):
                raise ValueError(
                    f"4D grid should have dimensions ('time', 'layer', 'y', 'x'). "
                    f"Instead, got {da.dims} for {da.name} in the "
                    f"{self.__class__.__name__} package. "
                )
        else:
            raise ValueError(
                f"Exceeded accepted amount of dimensions for "
                f"for {da.name} in the "
                f"{self.__class__.__name__} package. "
                f"Got {da.dims}. Can be at max ('time', 'layer', 'y', 'x')."
            )

    def _check_zero_dims(self):
        """
        Check if dim occurs with size zero. Sometimes xarray broadcasts empty
        dimensions. It also is possible to create DataArrays with a dim of size
        zero.
        """
        for dim in self.dataset.dims:
            if self.dataset.dims[dim] == 0:
                raise ValueError(
                    f"Provided dimension {dim} in {self.__class__.__name__} with size 0"
                )

    def _check_nan_consistent(self):
        """
        Check that not some cells in a grid of one var have ``np.nan``, whereas
        cells in another grid have data defined in that cell.
        """
        variables = self._get_vars_to_check()

        ds = self.dataset[variables]

        # Advanced Boundary Conditions can contain a mix of static data and
        # transient data (with a time dimension). ds.to_stacked_array cannot
        # handle this, therefore select the first timestep.
        if isinstance(self, AdvancedBoundaryCondition) and ("time" in ds.dims):
            ds = ds.isel(time=0)

        dims = list(ds.dims.keys())
        stacked = ds.to_stacked_array("var_dim", dims, name="stacked")
        n_nan_variables = np.isnan(stacked).sum(dim="var_dim")
        inconsistent_nan = (n_nan_variables > 0) & (n_nan_variables < len(variables))

        if inconsistent_nan.any():
            raise ValueError(
                f"Detected inconsistent data in {self.__class__.__name__}, "
                f"some variables contain nan, but others do not."
            )

    def _pkgcheck(self):
        super()._pkgcheck()

        self._check_zero_dims()
        self._check_all_nan()
        self._check_nan_consistent()


class AdvancedBoundaryCondition(BoundaryCondition, abc.ABC):
    """
    Class dedicated to advanced boundary conditions, since MF6 does not support
    binary files for Advanced Boundary conditions.

    The advanced boundary condition packages are: "uzf", "lak", "maw", "sfr".

    """

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
        outpath = directory / pkgname / f"{self._pkg_id}-pkgdata.dat"
        outpath.parent.mkdir(exist_ok=True, parents=True)
        package_data = self._package_data_to_sparse()
        self._write_file(outpath, package_data)

    def write(self, directory, pkgname, globaltimes, binary):
        self.fill_stress_perioddata()
        self.write_blockfile(directory, pkgname, globaltimes, binary=False)
        self.write_perioddata(directory, pkgname, binary=False)
        self.write_packagedata(directory, pkgname, binary=False)
