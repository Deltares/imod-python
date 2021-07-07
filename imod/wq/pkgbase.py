import abc
import pathlib
import warnings

import jinja2
import joblib
import numpy as np
import pandas as pd
import xarray as xr

import imod
from imod import util
from imod.wq import timeutil

from .caching import caching


def monkeypatch_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func

    return decorator


class Package(abc.ABC):
    """
    Base package for the different SEAWAT packages.
    Every package contains a ``_pkg_id`` for identification.
    Used to check for duplicate entries, or to group multiple systems together
    (riv, ghb, drn).

    The ``_template`` attribute is the template for a section of the runfile.
    This is filled in based on the metadata from the DataArrays that are within
    the Package.

    The ``_keywords`` attribute is a dictionary that's used to replace
    keyword argument by integer arguments for SEAWAT.
    """

    @classmethod
    def from_file(cls, path, cache_path=None, cache_verbose=0, **kwargs):
        """
        Loads an imod-wq package from a file (currently only netcdf is supported).

        This enables caching of intermediate input and should result in much
        faster model.write() times. To enable caching, provide a path to a
        ``joblib.Memory`` caching directory.

        Parameters
        ----------
        path : str, pathlib.Path
            Path to the file.
        cache_path : str, pathlib.Path, optional
            The path to the ``joblib.Memory`` caching dir where intermediate answers are stored.
        cache_verbose : int
            Verbosity flag of ``joblib.Memory``, controls the debug messages that are issued as
            functions are evaluated.
        **kwargs : keyword arguments
            Arbitrary keyword arguments forwarded to ``xarray.open_dataset()``, or
            ``xarray.open_zarr()``.
        Refer to the examples.

        Returns
        -------
        package : imod.wq.Package, imod.wq.CachingPackage
            Returns a package with data loaded from file. Returns a CachingPackage
            if a path to a ``joblib.Memory`` caching directory has been provided for ``cache``.

        Examples
        --------

        To load a package from a file, e.g. a River package:

        >>> river = imod.wq.River.from_file("river.nc")

        To load a package, and enable caching:

        >>> cache = "./.cache_dir"
        >>> river = imod.wq.River.from_file("river.nc", cache)

        For large datasets, you likely want to process it in chunks. You can
        forward keyword arguments to ``xarray.open_dataset()`` or
        ``xarray.open_zarr()``:

        >>> cache = "./.cache_dir"
        >>> river = imod.wq.River.from_file("river.nc", cache, chunks={"time": 1})

        Refer to the xarray documentation for the possible keyword arguments.
        """
        path = pathlib.Path(path)

        if path.suffix in (".zip", ".zarr"):
            # TODO: seems like a bug? Remove str() call if fixed in xarray/zarr
            cls._dataset = xr.open_zarr(str(path), **kwargs)
        else:
            cls._dataset = xr.open_dataset(path, **kwargs)

        # Monkeypatch class setitem to prevent costly dropna for all-nan layers at initialisation
        # https://mail.python.org/pipermail/python-dev/2008-January/076194.html
        # Note that the 'automagical' layer check does not work for 'from_file' packages
        # Within try-finally to make sure the original is always restored
        _org_setitem = cls.__setitem__
        try:

            @monkeypatch_method(cls)
            def __setitem__(self, key, value):
                super(__class__, self).__setitem__(key, value)

            pkg_kwargs = {var: cls._dataset[var] for var in cls._dataset.data_vars}
            if cache_path is None:
                return_cls = cls(**pkg_kwargs)
            else:
                # Dynamically construct a CachingPackage
                # Note:
                #    "a method cannot be decorated at class definition, because when
                #    the class is instantiated, the first argument (self) is bound,
                #    and no longer accessible to the Memory object."
                # See: https://joblib.readthedocs.io/en/latest/memory.html
                cache_path = pathlib.Path(cache_path)
                cache = joblib.Memory(cache_path, verbose=cache_verbose)
                CachingPackage = caching(cls, cache)
                return_cls = CachingPackage(path, **pkg_kwargs)

        finally:
            # Monkeypatch setitem back to what it was
            @monkeypatch_method(cls)
            def __setitem__(self, key, value):
                return _org_setitem(self, key, value)

        return return_cls

    def __init__(self):
        self.dataset = xr.Dataset()

    def __setitem__(self, key, value):
        if isinstance(value, xr.DataArray):
            if "z" in value.dims:
                if "layer" not in value.coords:
                    raise ValueError(
                        'Coordinate "layer" must be present in DataArrays with a "z" dimension'
                    )
                value = value.swap_dims({"z": "layer"})
            if "layer" in value.dims:
                value = value.dropna(dim="layer", how="all")
        self.dataset.__setitem__(key, value)

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def isel(self):
        raise NotImplementedError(
            f"Selection on packages not yet supported. "
            f"To make a selection on the xr.Dataset, call {self._pkg_id}.dataset.isel instead. "
            f"You can create a new package with a selection by calling {__class__.__name__}(**{self._pkg_id}.dataset.isel(**selection))"
        )

    def sel(self):
        raise NotImplementedError(
            f"Selection on packages not yet supported. "
            f"To make a selection on the xr.Dataset, call {self._pkg_id}.dataset.sel instead. "
            f"You can create a new package with a selection by calling {__class__.__name__}(**{self._pkg_id}.dataset.sel(**selection))"
        )

    def _replace_keyword(self, d, key):
        """
        Method to replace a readable keyword value by the corresponding cryptic
        integer value that SEAWAT demands.

        Dict ``d`` is updated in place.

        Parameters
        ----------
        d : dict
            Updated in place.
        key : str
            key of value in dict ``d`` to replace.
        """
        keyword = d[key][()]  # Get value from 0d np.array
        value = self._keywords[key][keyword]
        d[key] = value

    def _render(self, *args, **kwargs):
        """
        Rendering method for simple keyword packages (vdf, pcg).

        Returns
        -------
        rendered : str
            The rendered runfile part for a single boundary condition system.
        """
        d = {
            k: v.values for k, v in self.dataset.data_vars.items()
        }  # pylint: disable=no-member
        if hasattr(self, "_keywords"):
            for key in self._keywords.keys():
                self._replace_keyword(d, key)
        return self._template.format(**d)

    def _compose(self, d, pattern=None):
        # d : dict
        # pattern : string or re.pattern
        return util.compose(d, pattern).as_posix()

    def _compress_idflayers(self, values, range_path):
        """
        Compresses explicit layers into ranges

        Saves on number of lines, makes the runfile smaller, and imod-wq faster.
        """
        layers = np.array(list(values.keys()))
        if len(layers) == 1:
            return values

        breaks = np.argwhere(np.diff(layers) != 1)
        if breaks.size == 0:
            start = layers[0]
            end = layers[-1]
            return {f"{start}:{end}": range_path}
        starts = [0] + list(breaks[:, 0] + 1)
        ends = list(breaks[:, 0]) + [len(layers) - 1]

        compressed = {}
        for start_index, end_index in zip(starts, ends):
            start = layers[start_index]
            end = layers[end_index]
            if start == end:
                compressed[f"{start}"] = values[start]
            else:
                compressed[f"{start}:{end}"] = range_path

        return compressed

    def _compose_values_layer(
        self, varname, directory, nlayer, time=None, da=None, compress=True
    ):
        """
        Composes paths to files, or gets the appropriate scalar value for
        a single variable in a dataset.

        Parameters
        ----------
        varname : str
            variable name of the DataArray
        directory : str
            Path to working directory, where files will be written.
            Necessary to generate the paths for the runfile.
        time : datetime like, optional
            Time corresponding to the value.
        da : xr.DataArray, optional
            In some cases fetching the DataArray by varname is not desired.
            It can be passed directly via this optional argument.
        compress : boolean
            Whether or not to compress the layers using the imod-wq macros.
            Should be disabled for time-dependent input.

        Returns
        -------
        values : dict
            Dictionary containing the {layer number: path to file}.
            Alternatively: {layer number: scalar value}. The layer number may be
            a wildcard (e.g. '?').
        """
        pattern = "{name}"

        values = {}
        if da is None:
            da = self.dataset[varname]

        d = {"directory": directory, "name": varname, "extension": ".idf"}

        if "species" in da.coords:
            d["species"] = da.coords["species"].values
            pattern += "_c{species}"

        if time is not None:
            d["time"] = time
            pattern += "_{time:%Y%m%d%H%M%S}"

        # Scalar value or not?
        # If it's a scalar value we can immediately write
        # otherwise, we have to write a file path
        if "y" not in da.coords and "x" not in da.coords:
            idf = False
        else:
            idf = True

        if "layer" not in da.coords:
            if idf:
                # Special case concentration
                # Using "?" results in too many sinks and sources according to imod-wq.
                pattern += "{extension}"
                if hasattr(self, "_ssm_layers"):
                    for layer in self._ssm_layers:
                        d["layer"] = layer
                        values[layer] = self._compose(d, pattern=pattern)
                else:
                    values["?"] = self._compose(d, pattern=pattern)
            else:
                # Special case concentration
                # Using "?" results in too many sinks and sources according to imod-wq.
                if hasattr(self, "_ssm_layers"):
                    for layer in self._ssm_layers:
                        values[layer] = da.values[()]
                    values = self._compress_values(values)
                else:
                    values["?"] = da.values[()]

        else:
            pattern += "_l{layer}{extension}"
            for layer in np.atleast_1d(da.coords["layer"].values):
                if idf:
                    d["layer"] = layer
                    values[layer] = self._compose(d, pattern=pattern)
                else:
                    if "layer" in da.dims:
                        values[layer] = da.sel(layer=layer).values[()]
                    else:
                        values[layer] = da.values[()]

        # Compress the runfile contents using the imod-wq macros
        if "layer" in da.dims:
            if idf and da["layer"].size == nlayer:
                # Compose does not accept non-integers, so use 0, then replace
                d["layer"] = 0
                token_path = util.compose(d, pattern=pattern).as_posix()
                token_path = token_path.replace("_l0", "_l$")
                values = {"$": token_path}
            elif idf and compress:
                # Compose does not accept non-integers, so use 0, then replace
                d["layer"] = 0
                range_path = util.compose(d, pattern=pattern).as_posix()
                range_path = range_path.replace("_l0", "_l:")
                values = self._compress_idflayers(values, range_path)
            elif compress:
                values = self._compress_values(values)

        return values

    def _compress_values(self, values):
        """
        Compress repeated values into fewer lines, aimed at imod-wq macros.
        """
        keys = list(values.keys())
        values = np.array(list(values.values()))
        n = len(values)
        # Try fast method first
        try:
            index_ends = np.argwhere(np.diff(values) != 0)
        except np.core._exceptions.UFuncTypeError:
            # Now try a fully general method
            index_ends = []
            for i in range(n - 1):
                if values[i] != values[i + 1]:
                    index_ends.append(i)
            index_ends = np.array(index_ends)

        index_ends = np.append(index_ends, n - 1)
        index_starts = np.insert(index_ends[:-1] + 1, 0, 0)
        compressed = {}
        for start_index, end_index in zip(index_starts, index_ends):
            s = keys[start_index]
            e = keys[end_index]
            value = values[start_index]
            if s == e:
                compressed[f"{s}"] = value
            else:
                compressed[f"{s}:{e}"] = value
        return compressed

    def _compose_values_time(self, varname, globaltimes):
        da = self.dataset[varname]
        values = {}

        if "time" in da.coords:
            package_times = da.coords["time"].values

            starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)
            for itime, start_end in enumerate(starts_ends):
                # TODO: this now fails on a non-dim time too
                # solution roughly the same as for layer above?
                values[start_end] = da.isel(time=itime).values[()]
        else:
            values["?"] = da.values[()]

        values = self._compress_values(values)
        return values

    def save(self, directory):
        for name, da in self.dataset.data_vars.items():  # pylint: disable=no-member
            if "y" in da.coords and "x" in da.coords:
                path = pathlib.Path(directory).joinpath(name)
                imod.idf.save(path, da)

    def _check_positive(self, varnames):
        for var in varnames:
            # Take care with nan values
            if (self.dataset[var] < 0).any():
                raise ValueError(f"{var} in {self} must be positive")

    def _check_range(self, varname, lower, upper):
        # TODO: this isn't used anywhere so far.
        warn = False
        msg = ""
        if (self.dataset[varname] < lower).any():
            warn = True
            msg += f"{varname} in {self}: values lower than {lower} detected. "
        if (self.dataset[varname] > upper).any():
            warn = True
            msg += f"{varname} in {self}: values higher than {upper} detected."
        if warn:
            warnings.warn(msg, RuntimeWarning)

    def _check_location_consistent(self, varnames):
        dims = set(self.dataset.dims)
        is_scalar = {}
        for var in varnames:
            scalar = (self.dataset[var].shape == ()) or not any(
                dim in self.dataset[var].dims for dim in ["time", "layer", "y", "x"]
            )
            if not scalar:  # skip scalar value
                dims = dims.intersection(self.dataset[var].dims)
            is_scalar[var] = scalar

        is_nan = True
        for var in varnames:
            if not is_scalar[var]:  # skip scalar values
                # dimensions cannot change for in-place operations
                # reduce to lowest set of dimension (e.g. just x and y)
                var_dims = set(self.dataset[var].dims)
                reduce_dims = var_dims.difference(dims)
                # Inplace boolean operator
                is_nan &= np.isnan(self.dataset[var]).all(dim=reduce_dims)

        for var in varnames:
            if not is_scalar[var]:  # skip scalar values
                if (np.isnan(self.dataset[var]) ^ is_nan).any():
                    raise ValueError(
                        f"{var} in {self} is not consistent with all variables in: "
                        f"{', '.join(varnames)}. nan values do not line up."
                    )

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


class BoundaryCondition(Package, abc.ABC):
    """
    Base package for (transient) boundary conditions:
    * recharge
    * general head boundary
    * constant head
    * river
    * drainage
    """

    _template = jinja2.Template(
        "    {%- for name, dictname in mapping -%}"
        "        {%- for time, timedict in dicts[dictname].items() -%}"
        "            {%- for layer, value in timedict.items() %}\n"
        "    {{name}}_p{{time}}_s{{system_index}}_l{{layer}} = {{value}}\n"
        "            {%- endfor -%}\n"
        "        {%- endfor -%}"
        "    {%- endfor -%}"
    )
    _ssm_template = jinja2.Template(
        "{%- for species, timedict in concentration.items() -%}"
        "    {%- for time, layerdict in timedict.items() -%}"
        "       {%- for layer, value in layerdict.items() %}\n"
        "    c{{pkg_id}}_t{{species}}_p{{time}}_l{{layer}} = {{value}}\n"
        "        {%- endfor -%}"
        "    {%- endfor -%}"
        "{%- endfor -%}"
    )

    def add_timemap(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "add_timemap is deprecated: use repeat_stress instead", FutureWarning
        )
        self.repeat_stress(*args, **kwargs)

    def _repeat_stress(self, varname, value, use_cftime):
        if value is not None:
            if varname not in self.dataset.data_vars:
                raise ValueError(
                    f"{varname} does not occur in {self}\n cannot repeat stress"
                )
            if "time" not in self.dataset[varname].coords:
                raise ValueError(
                    f"{varname} in {self}\n does not have dimension time, cannot repeat stress."
                )

            # Replace both key and value by the right datetime type
            d = {
                timeutil.to_datetime(k, use_cftime): timeutil.to_datetime(v, use_cftime)
                for k, v in value.items()
            }
            self.dataset[varname].attrs["stress_repeats"] = d

    def _compose_values_timelayer(
        self, varname, globaltimes, directory, nlayer, da=None
    ):
        """
        Composes paths to files, or gets the appropriate scalar value for
        a single variable in a dataset.

        Parameters
        ----------
        varname : str
            variable name of the DataArray
        globaltimes : list, np.array
            Holds the global times, i.e. the combined unique times of
            every boundary condition that are used to define the stress
            periods.
            The times of the BoundaryCondition do not have to match all
            the global times. When a globaltime is not present in the
            BoundaryCondition, the value of the first previous available time is
            filled in. The effective result is a forward fill in time.
        directory : str
            Path to working directory, where files will be written.
            Necessary to generate the paths for the runfile.
        da : xr.DataArray, optional
            In some cases fetching the DataArray by varname is not desired.
            It can be passed directly via this optional argument.

        Returns
        -------
        values : dict
            Dictionary containing the {stress period number: {layer number: path
            to file}}. Alternatively: {stress period number: {layer number: scalar
            value}}.
            The stress period number and layer number may be wildcards (e.g. '?').
        """

        values = {}

        if da is None:
            da = self.dataset[varname]

        if "time" in da.coords:
            da_times = da.coords["time"].values
            if "stress_repeats" in da.attrs:
                stress_repeats_keys = np.array(list(da.attrs["stress_repeats"].keys()))
                stress_repeats_values = np.array(
                    list(da.attrs["stress_repeats"].values())
                )
                package_times, inds = np.unique(
                    np.concatenate([da_times, stress_repeats_keys]), return_index=True
                )
                # Times to write in the runfile
                runfile_times = np.concatenate([da_times, stress_repeats_values])[inds]
            else:
                runfile_times = package_times = da_times

            starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)

            for time, start_end in zip(runfile_times, starts_ends):
                # Check whether any range occurs in the input.
                # If does does, compress should be False
                compress = not (":" in start_end)
                values[start_end] = self._compose_values_layer(
                    varname,
                    directory,
                    nlayer=nlayer,
                    time=time,
                    da=da,
                    compress=compress,
                )

        else:
            values["?"] = self._compose_values_layer(
                varname, directory, nlayer=nlayer, da=da
            )

        return values

    def _max_active_n(self, varname, nlayer, nrow, ncol):
        """
        Determine the maximum active number of cells that are active
        during a stress period.

        Parameters
        ----------
        varname : str
            name of the variable to use to calculate the maximum number of
            active cells. Generally conductance.
        shape : tuple of ints
            nlay, nrow, ncol taken from ibound.
        """
        # First compute active number of cells
        if "time" in self.dataset[varname].coords:
            nmax = int(self.dataset[varname].groupby("time").count(xr.ALL_DIMS).max())
        else:
            nmax = int(self.dataset[varname].count())
        if not "layer" in self.dataset.coords:  # Then it applies to every layer
            nmax *= nlayer
        self._cellcount = nmax  # Store cellcount so it can be re-used for ssm.
        self._ssm_cellcount = nmax

        # Second, compute active number of sinks and sources
        # overwite _ssm_cellcount if more specific info is available.
        if "concentration" in self.dataset.data_vars:
            da = self["concentration"]

            if "species" in da.coords:
                nspecies = da.coords["species"].size
            else:
                nspecies = 1

            if "y" not in da.coords and "x" not in da.coords:
                # It's not idf data, but scalar instead
                if "layer" in self.dataset.coords:
                    # Store layers for rendering
                    da_nlayer = self.dataset.coords["layer"].size
                    if da_nlayer == nlayer:
                        # Insert wildcard
                        self._ssm_layers = ["?"]
                    else:
                        self._ssm_layers = self.dataset.coords["layer"].values
                        nlayer = da_nlayer
                # Sinks and sources are applied everywhere
                # in contrast to other inputs
                nmax = nlayer * nrow * ncol

            self._ssm_cellcount = nmax * nspecies

        return nmax

    def _render(self, directory, globaltimes, system_index, nlayer):
        """
        Parameters
        ----------
        directory : str
            Path to working directory, where files will be written.
            Necessary to generate the paths for the runfile.
        globaltimes : list, np.array
            Holds the global times, i.e. the combined unique times of
            every boundary condition that are used to define the stress
            periods.
        system_index : int
            Drainage, River, and GeneralHeadBoundary support multiple systems.
            This is the number that ends up in the runfile for a given
            BoundaryCondition object.
            Note that MT3DMS does not fully support multiple systems, and only
            the first system can act as source of species.

        Returns
        -------
        rendered : str
            The rendered runfile part for a single boundary condition system.
        """
        mapping = tuple(
            [(k, v) for k, v in self._mapping if v in self.dataset.data_vars]
        )
        d = {"mapping": mapping, "system_index": system_index}
        dicts = {}

        for varname in self.dataset.data_vars.keys():  # pylint: disable=no-member
            if varname == "concentration":
                continue
            dicts[varname] = self._compose_values_timelayer(
                varname, globaltimes, directory, nlayer=nlayer
            )

        d["dicts"] = dicts

        return self._template.render(d)

    def _render_ssm(self, directory, globaltimes, nlayer):
        """
        Parameters
        ----------
        directory : str
            Path to working directory, where files will be written.
            Necessary to generate the paths for the runfile.
        globaltimes : list, np.array
            Holds the global times, i.e. the combined unique times of
            every boundary condition that are used to define the stress
            periods.

        Returns
        -------
        rendered : str
            The rendered runfile SSM part for a single boundary condition system.
        """

        if "concentration" not in self.dataset.data_vars:
            return ""

        d = {"pkg_id": self._pkg_id}
        if "species" in self.dataset["concentration"].coords:
            concentration = {}
            for species in self.dataset["concentration"]["species"].values:
                concentration[species] = self._compose_values_timelayer(
                    varname="concentration",
                    da=self.dataset["concentration"].sel(species=species),
                    globaltimes=globaltimes,
                    directory=directory,
                    nlayer=nlayer,
                )
        else:
            concentration = {
                1: self._compose_values_timelayer(
                    varname="concentration",
                    da=self.dataset["concentration"],
                    globaltimes=globaltimes,
                    directory=directory,
                    nlayer=nlayer,
                )
            }
        d["concentration"] = concentration

        return self._ssm_template.render(d)
