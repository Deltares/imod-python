import abc
import numbers
import pathlib
from collections import defaultdict
from typing import Any, Dict, List

import cftime
import jinja2
import numpy as np
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.validation import validation_pkg_error_message
from imod.schemata import ValidationError

TRANSPORT_PACKAGES = ("adv", "dsp", "ssm", "mst", "ist", "src")


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
    if notnull.ndim == 1 and layer.size == 1:
        recarr["cell2d"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, icell2d = np.argwhere(notnull).transpose()
        recarr["cell2d"] = icell2d + 1
        recarr["layer"] = layer[ilayer]
    return recarr


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
        path = pathlib.Path(path)
        if path.suffix in (".zip", ".zarr"):
            # TODO: seems like a bug? Remove str() call if fixed in xarray/zarr
            dataset = xr.open_zarr(str(path), **kwargs)
        else:
            dataset = xr.open_dataset(path, **kwargs)

        if dataset.ugrid_roles.topology:
            dataset = xu.UgridDataset(dataset)
            dataset = imod.util.from_mdal_compliant_ugrid2d(dataset)

        # Replace NaNs by None
        for key, value in dataset.items():
            stripped_value = value.values[()]
            if isinstance(stripped_value, numbers.Real) and np.isnan(stripped_value):
                dataset[key] = None

        instance = cls.__new__(cls)
        instance.dataset = dataset
        return instance

    def __init__(self, allargs=None):
        if allargs is not None:
            for arg in allargs.values():
                if isinstance(arg, xu.UgridDataArray):
                    self.dataset = xu.UgridDataset(grids=arg.ugrid.grid)
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
        # When dumping to netCDF and reading back, None will have been
        # converted into a NaN. Only check NaN if it's a floating type to avoid
        # TypeErrors.
        elif np.issubdtype(type(value), np.floating) and np.isnan(value):
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
            auxiliary_vars = (
                self.get_auxiliary_variable_names()
            )  # returns something like {"concentration": "species"}
            if datavar in auxiliary_vars.keys():  # if datavar is concentration
                if (
                    auxiliary_vars[datavar] in ds[datavar].dims
                ):  # if this concentration array has the species dimension
                    for s in ds[datavar].values:  # loop over species
                        arrdict[s] = (
                            ds[datavar]
                            .sel({auxiliary_vars[datavar]: s})
                            .values  # store species array under its species name
                        )
            else:
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
            data = da.values
            if data.ndim > 2:
                np.savetxt(fname=f, X=da.values.reshape((1, -1)), fmt=fmt)
            else:
                np.savetxt(fname=f, X=da.values, fmt=fmt)

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

    def _validate(self, schemata: Dict, **kwargs) -> Dict[str, List[ValidationError]]:
        errors = defaultdict(list)
        for variable, var_schemata in schemata.items():
            for schema in var_schemata:
                if (
                    variable in self.dataset.keys()
                ):  # concentration only added to dataset if specified
                    try:
                        schema.validate(self.dataset[variable], **kwargs)
                    except ValidationError as e:
                        errors[variable].append(e)
        return errors

    def _validate_init_schemata(self, validate: bool):
        """
        Run the "cheap" schema validations.

        The expensive validations are run during writing. Some are only
        available then: e.g. idomain to determine active part of domain.
        """
        if not validate:
            return
        errors = self._validate(self._init_schemata)
        if len(errors) > 0:
            message = validation_pkg_error_message(errors)
            raise ValidationError(message)
        return

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

    def period_data(self):
        result = []
        if hasattr(self, "_period_data"):
            result += self._period_data
        if hasattr(self, "_auxiliary_data"):
            for aux_var_name, aux_var_dimensions in self._auxiliary_data.items():
                if aux_var_name in self.dataset.keys():
                    for s in (
                        self.dataset[aux_var_name].coords[aux_var_dimensions].values
                    ):
                        result.append(s)
        return result

    def add_periodic_auxiliary_variable(self):
        if hasattr(self, "_auxiliary_data"):
            for aux_var_name, aux_var_dimensions in self._auxiliary_data.items():
                aux_coords = (
                    self.dataset[aux_var_name].coords[aux_var_dimensions].values
                )
                for s in aux_coords:
                    self.dataset[s] = self.dataset[aux_var_name].sel(
                        {aux_var_dimensions: s}
                    )

    def get_auxiliary_variable_names(self):
        result = {}
        if hasattr(self, "_auxiliary_data"):
            result.update(self._auxiliary_data)
        return result

    def copy(self) -> Any:
        # All state should be contained in the dataset.
        return type(self)(**self.dataset.copy())

    def clip_domain(
        self, layer: slice = None, x: slice = None, y: slice = None
    ) -> "Package":
        """
        Clip a variable along the specified dimensions.

        Accepts only start and end dimensions, provided as ``slice``.

        Parameters
        ----------
        layer: slice
        x: slice
        y: slice

        Returns
        -------
        clipped : Package
        """

        def check_if_slice(key, value):
            if value is None:
                return slice(None, None)
            elif not isinstance(value, slice):
                raise TypeError(
                    f"Expected slice for {key}. " f"Received: {type(value).__name__}"
                )
            return value

        layer = check_if_slice("layer", layer)
        x = check_if_slice("x", x)
        y = check_if_slice("y", y)

        if isinstance(self.dataset, xu.UgridDataset):
            clipped = self.dataset.ugrid.sel(x=x, y=y).sel(layer=layer)
        else:
            clipped = self.dataset.sel(layer=layer, x=x, y=y)

        return clipped

    def mask(self, domain: xr.DataArray) -> Any:
        """
        Mask values outside of domain.

        Floating values outside of the condition are set to NaN (nodata).
        Integer values outside of the condition are set to 0 (inactive in
        MODFLOW terms).

        Parameters
        ----------
        domain: xr.DataArray of bools
            The condition. Preserve values where True, discard where False.

        Returns
        -------
        masked: Package
            The package with part masked.
        """
        masked = {}
        for var, da in self.dataset.data_vars.items():
            if set(domain.dims).issubset(da.dims):
                # Check if this should be: np.issubdtype(da.dtype, np.floating)
                if issubclass(da.dtype, numbers.Real):
                    masked[var] = da.where(domain, other=np.nan)
                elif issubclass(da.dtype, numbers.Integral):
                    masked[var] = da.where(domain, other=0)
                else:
                    raise TypeError(
                        f"Expected dtype float or integer. Received instead: {da.dtype}"
                    )
            else:
                masked[var] = da

        return type(self)(**masked)


class BoundaryCondition(Package, abc.ABC):
    """
    BoundaryCondition is used to share methods for specific stress packages
    with a time component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    This class only supports `list input
    <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=19>`_,
    not the array input which is used in :class:`Package`.
    """

    def set_repeat_stress(self, times) -> None:
        """
        Set repeat stresses: re-use data of earlier periods.

        Parameters
        ----------
        times: Dict of datetime-like to datetime-like.
            The data of the value datetime is used for the key datetime.
        """
        keys = [
            imod.wq.timeutil.to_datetime(key, use_cftime=False) for key in times.keys()
        ]
        values = [
            imod.wq.timeutil.to_datetime(value, use_cftime=False)
            for value in times.values()
        ]
        self.dataset["repeat_stress"] = xr.DataArray(
            data=np.column_stack((keys, values)),
            dims=("repeat", "repeat_items"),
        )

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

    def assign_dims(self, arg) -> Dict:
        is_da = isinstance(arg, xr.DataArray)
        if is_da and "time" in arg.coords:
            if arg.ndim != 2:
                raise ValueError("time varying variable: must be 2d")
            if arg.dims[0] != "time":
                arg = arg.transpose()
            da = xr.DataArray(
                data=arg.values, coords={"time": arg["time"]}, dims=["time", "index"]
            )
            return da
        elif is_da:
            return ("index", arg.values)
        else:
            return ("index", arg)

    def clip_domain(
        self,
        time: slice = None,
        layer: slice = None,
        x: slice = None,
        y: slice = None,
    ) -> "BoundaryCondition":
        """
        Clip a variable along the specified dimensions.

        Accepts only start and end dimensions, provided as ``slice``.

        Parameters
        ----------
        time: slice
        layer: slice
        x: slice
        y: slice

        Returns
        -------
        clipped : BoundaryCondition
        """
        if "time" in self.dataset.coords and time is not None:
            if not isinstance(time, slice):
                raise TypeError(
                    f"Expected slice for time, " f"received: {type(time).__name__}"
                )

            use_cftime = isinstance(self.dataset["time"][0], cftime.datetime)
            selection = self.dataset.sel(time=time)
            # time_sel.start included? Otherwise, concat
            if (
                imod.wq.timeutil.to_datetime(time.start, use_cftime)
                not in selection.time
            ):
                start = self.dataset.sel(time=time.start, method="ffill")
                # Is this required?
                # start["time"] = imod.wq.timeutil.to_datetime(start, use_cftime)
                selection = xr.concat(
                    [start, selection], dim="time", data_vars="minimal"
                )
            # TODO: bfill
        else:
            selection = self.dataset

        return super(selection, selection).clip(layer=layer, x=x, y=y)


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


class DisStructuredBoundaryCondition(BoundaryCondition):
    def to_sparse(self, arrdict, layer):
        spec = []
        for key in arrdict:
            if key in ["layer", "row", "column"]:
                spec.append((key, np.int32))
            else:
                spec.append((key, np.float64))

        sparse_dtype = np.dtype(spec)
        nrow = next(iter(arrdict.values())).size
        recarr = np.empty(nrow, dtype=sparse_dtype)
        for key, arr in arrdict.items():
            recarr[key] = arr
        return recarr


class DisVerticesBoundaryCondition(BoundaryCondition):
    def to_sparse(self, arrdict, layer):
        spec = []
        for key in arrdict:
            if key in ["layer", "cell2d"]:
                spec.append((key, np.int32))
            else:
                spec.append((key, np.float64))

        sparse_dtype = np.dtype(spec)
        nrow = next(iter(arrdict.values())).size
        recarr = np.empty(nrow, dtype=sparse_dtype)
        for key, arr in arrdict.items():
            recarr[key] = arr
        return recarr
