import abc
import pathlib

import cftime
import jinja2
import joblib
import numpy as np
import xarray as xr

import imod
from imod import util
from imod.flow import timeutil
from imod.wq import caching


class Vividict(dict):
    """
    Vividict is used to generate tree structures
    Source: https://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries/19829714#19829714
    """

    def __missing__(self, key):
        value = self[key] = type(self)()  # retain local pointer to value
        return value  # faster to return than dict lookup


class Package(
    abc.ABC
):  # TODO: Abstract base class really necessary? Are we using abstract methods?
    """
    Base package for the different iMODFLOW packages.  Package is used to share
    methods for specific packages with no time component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    Every package contains a ``_pkg_id`` for identification.  Used to check for
    duplicate entries, or to group multiple systems together (riv, ghb, drn).

    The ``_template_runfile`` attribute is the template for a section of the
    runfile.  This is filled in based on the metadata from the DataArrays that
    are within the Package. Same applies to ``_template_projectfile`` for the
    projectfile.
    """

    __slots__ = ("_pkg_id", "_variable_order", "dataset")

    # TODO Runfile template not implemented yet
    _template_runfile = jinja2.Template(
        "{%- for layer, path in variable_data %}\n"
        '{{layer}}, 1.0, 0.0, "{{path}}"\n'
        "{%- endfor %}\n"
    )

    _template_projectfile = jinja2.Template(
        "0001, ({{pkg_id}}), 1, {{name}}, {{variable_order}}\n"
        '{{"{:03d}".format(variable_order|length)}}, {{"{:03d}".format(n_entry)}}\n'
        "{%- for variable in variable_order%}\n"  # Preserve variable order
        "{%-    for layer, value in package_data[variable].items()%}\n"
        "{%-        if value is string %}\n"
        # If string then assume path:
        # 1 indicates the layer is activated
        # 2 indicates the second element of the final two elements should be read
        # 1.000 is the multiplication factor
        # 0.000 is the addition factor
        # -9999 indicates there is no data, following iMOD usual practice
        '1, 2, {{"{:03d}".format(layer)}}, 1.000, 0.000, -9999., {{value}}\n'
        "{%-        else %}\n"
        # Else assume a constant value is provided
        '1, 1, {{"{:03d}".format(layer)}}, 1.000, 0.000, {{value}}, ""\n'
        "{%-        endif %}\n"
        "{%-    endfor %}\n"
        "{%- endfor %}\n"
    )

    def __init__(self):
        self.dataset = xr.Dataset()

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

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

        pkg_kwargs = {var: cls._dataset[var] for var in cls._dataset.data_vars}
        if cache_path is None:
            return cls(**pkg_kwargs)
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
            return CachingPackage(path, **pkg_kwargs)

    # TODO:
    # def __getattribute__(self, name):
    # "implement the: https://github.com/xgcm/xgcm/issues/225#issuecomment-762248339"
    #    pass

    def _pkgcheck(self, **kwargs):
        pass

    def _check_if_nan_in_active_cells(self, active_cells=None, vars_to_check=None):
        """Check if there are any nans in the active domain"""
        for var in vars_to_check:
            if (active_cells & np.isnan(self.dataset[var])).any():
                raise ValueError(
                    f"Active cells in ibound may not have a nan value in {var}"
                )

    def _check_positive(self, varnames):
        for var in varnames:
            # Take care with nan values
            if (self[var] < 0).any():
                raise ValueError(f"{var} in {self} must be positive")

    def _is_periodic(self):
        # Periodic stresses are defined for all variables
        first_var = list(self.dataset.data_vars)[0]
        return "stress_periodic" in self.dataset[first_var].attrs

    def _hastime(self):
        return (self._pkg_id == "wel" and "time" in self.dataset) or (
            "time" in self.dataset.coords
        )

    def compose(self, directory, globaltimes, nlayer, composition=None, **ignored):
        """
        Composes package, not useful for boundary conditions

        Parameters
        ----------
        directory : str
            Path to working directory, where files will be written.
            Necessary to generate the paths for the runfile.
        globaltimes : list #TODO make this an *arg, change order.
            Not used, only included to comply with BoundaryCondition.compose
        composition : Vividict
            Existing composition to add composed packages to.
        **ignored
            Contains keyword arguments unused for packages, i.e. compose_projectfile
        """

        if composition is None:
            composition = Vividict()

        for varname in self._variable_order:
            composition[self._pkg_id][varname] = self._compose_values_layer(
                varname, directory, nlayer
            )

        return composition

    def _compose_path(self, d, pattern=None):
        """
        Construct a filename, following the iMOD conventions.
        Returns an absolute path.

        Parameters
        ----------
        d : dict
            dict of parts (time, layer) for filename.
        pattern : string or re.pattern
            Format to create pattern for.

        Returns
        -------
        str
            Absolute path.

        """
        return str(util.compose(d, pattern).resolve())

    def _compose_values_layer(self, varname, directory, nlayer, time=None):
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
        nlayer : int
            Amount of layers
        time : datetime like, optional
            Time corresponding to the value.

        Returns
        -------
        values : dict
            Dictionary containing the {layer number: path to file}.
            Alternatively: {layer number: scalar value}. The layer number may be
            a wildcard (e.g. '?').
        """
        pattern = "{name}"

        values = Vividict()
        da = self[varname]

        d = {"directory": directory, "name": varname, "extension": ".idf"}

        if time is not None:
            if isinstance(time, (np.datetime64, cftime.datetime)):
                d["time"] = time
                pattern += "_{time:%Y%m%d%H%M%S}"
            else:
                d["timestr"] = time
                pattern += "_{timestr}"

        # Scalar value or not?
        # If it's a scalar value we can immediately write
        # otherwise, we have to write a file path
        if "y" not in da.coords and "x" not in da.coords:
            idf = False
        else:
            idf = True

        if "layer" not in da.coords:
            if idf:
                pattern += "{extension}"
                for layer in range(1, nlayer + 1):  # 1-based indexing
                    values[layer] = self._compose_path(d, pattern=pattern)
            else:
                for layer in range(1, nlayer + 1):  # 1-based indexing
                    values[layer] = da.values[()]

        else:
            pattern += "_l{layer}{extension}"
            for layer in np.atleast_1d(da.coords["layer"].values):
                if idf:
                    d["layer"] = layer
                    values[layer] = self._compose_path(d, pattern=pattern)
                else:
                    if "layer" in da.dims:
                        values[layer] = da.sel(layer=layer).values[()]
                    else:
                        values[layer] = da.values[()]

        return values

    def _compose_values_time(self, varname, globaltimes):
        da = self.dataset[varname]
        values = {}

        if "time" in da.coords:
            package_times = da.coords["time"].values

            starts = timeutil.forcing_starts(package_times, globaltimes)
            for itime, start in enumerate(starts):
                # TODO: this now fails on a non-dim time too
                # solution roughly the same as for layer above?
                values[start] = da.isel(time=itime).values[()]
        else:
            values["steady-state"] = da.values[()]

        return values

    def _render_projectfile(self, **kwargs):
        """
        Returns
        -------
        rendered : str
            The rendered projfectfile part,
            if part of PkgGroup: for a single boundary condition system.
        """
        return self._template_projectfile.render(**kwargs)

    def _render_runfile(self, **kwargs):
        """
        Returns
        -------
        rendered : str
            The rendered runfile part,
            if part of PkgGroup: for a single boundary condition system.
        """
        return self._template.render(**kwargs)

    def save(self, directory):
        for name, da in self.dataset.data_vars.items():  # pylint: disable=no-member
            if "y" in da.coords and "x" in da.coords:
                path = pathlib.Path(directory).joinpath(name)
                imod.idf.save(path, da)


class BoundaryCondition(Package, abc.ABC):
    """
    BoundaryCondition is used to share methods for specific stress packages
    with a time component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.
    """

    _template_projectfile = jinja2.Template(
        # Specify amount of timesteps for a package
        # 1 indicates if package is active or not
        '{{"{:04d}".format(package_data|length)}}, ({{pkg_id}}), 1, {{name}}, {{variable_order}}\n'
        "{%- for time_key, time_data in package_data.items()%}\n"
        # Specify stress period
        # Specify amount of variables and entries(nlay, nsys) to be expected
        "{{times[time_key]}}\n"
        '{{"{:03d}".format(time_data|length)}}, {{"{:03d}".format(n_entry)}}\n'
        "{%-    for variable in variable_order%}\n"  # Preserve variable order
        "{%-        for system, system_data in time_data[variable].items() %}\n"
        "{%-            for layer, value in system_data.items() %}\n"
        "{%-                if value is string %}\n"
        # If string then assume path:
        # 1 indicates the layer is activated
        # 2 indicates the second element of the final two elements should be read
        # 1.000 is the multiplication factor
        # 0.000 is the addition factor
        # -9999 indicates there is no data, following iMOD usual practice
        '1, 2, {{"{:03d}".format(layer)}}, 1.000, 0.000, -9999., {{value}}\n'
        "{%-                else %}\n"
        # Else assume a constant value is provided
        '1, 1, {{"{:03d}".format(layer)}}, 1.000, 0.000, {{value}}, ""\n'
        "{%-                endif %}\n"
        "{%-            endfor %}\n"
        "{%-        endfor %}\n"
        "{%-    endfor %}\n"
        "{%- endfor %}\n"
    )

    def repeat_stress(self, use_cftime=False, **repeats):
        """
        Repeat stress periods.

        Parameters
        ----------
        use_cftime: bool
            Whether to force datetimes to cftime or not.
        **repeats: dict of datetime - datetime pairs
            keyword argument with variable name as keyword and
            a dict as value. This dict contains a datetime as key
            which maps to another already existing datetime in the
            BoundaryCondition, for which data should be repeated.

        """
        # This is a generic implementation of repeat_stress in iMOD-WQ.
        # Genericity in this module is possible because
        # of the existence of self._variable_order.

        # Check first if all the provided repeats are actually
        # arguments of the package
        self._varnames_in_variable_order(repeats.keys())

        # Loop over variable order
        for varname in self._variable_order:
            if varname in repeats.keys():
                self._repeat_stress(varname, repeats[varname], use_cftime=use_cftime)
            else:  # Default to None, like in WQ implementation
                self._repeat_stress(varname, None, use_cftime=use_cftime)

    def _repeat_stress(self, varname, value, use_cftime):
        if value is not None:
            if varname not in self.dataset:
                raise ValueError(
                    f"{varname} does not occur in {self}\n cannot add stress_repeats"
                )
            if "time" not in self[varname].coords:
                raise ValueError(
                    f"{varname} in {self}\n does not have dimension time, cannot add stress_repeats."
                )

            # Replace both key and value by the right datetime type
            d = {
                imod.wq.timeutil.to_datetime(
                    k, use_cftime
                ): imod.wq.timeutil.to_datetime(v, use_cftime)
                for k, v in value.items()
            }
            self[varname].attrs["stress_repeats"] = d

    def periodic_stress(
        self,
        periods,
        use_cftime=False,
    ):
        """
        Periodic stress periods.

        Adds periodic stresses to each variable in the package.  iMODFLOW will
        then implicitly repeat these.

        The iMOD manual currently states: 'A PERIOD repeats until another time
        definition is more close to the current time step'.

        Parameters
        ----------
        periods: dict of datetime - string pairs
            contains a datetime as key which maps to a period label.  This will
            be used for the entire package.
        use_cftime: bool
            Whether to force datetimes to cftime or not.

        Examples
        --------
        The following example assigns a higher head to the summer period than
        winter period.  iMODFLOW will switch to period "summer" once
        'xxxx-04-01' has passed, and "winter" once 'xxxx-10-01' has passed.

        >>> times = [np.datetime64('2000-04-01'), np.datetime64('2000-10-01')]

        >>> head_periodic = xr.DataArray([2., 1.], coords={"time": times}, dims=["time"])

        >>> timemap = {times[0]: "summer", times[1]: "winter"}

        >>> ghb = GeneralHeadBoundary(head = head_periodic, conductance = 10.)
        >>> ghb.periodic_stress(timemap)

        """

        if "time" not in self.dataset.coords:
            raise ValueError(
                f"{self} does not have dimension time, cannot add stress_periodic."
            )

        if self.dataset.coords["time"].size != len(periods):
            raise ValueError(
                f"{self} does not have the same amounnt of timesteps as number of periods."
            )

        # Replace both key and value by the right datetime type
        d = {imod.wq.timeutil.to_datetime(k, use_cftime): v for k, v in periods.items()}

        for varname in self._variable_order:
            self.dataset[varname].attrs["stress_periodic"] = d

    def _varnames_in_variable_order(self, varnames):
        """Check if varname in _variable_order"""
        for varname in varnames:
            if varname not in self._variable_order:
                raise ValueError(
                    f"{varname} not recognized for {self}, choose one of {self._variable_order}"
                )

    def _get_runfile_times(self, da, globaltimes, ds_times=None):
        if ds_times is None:
            ds_times = self.dataset.coords["time"].values

        if "stress_repeats" in da.attrs:
            stress_repeats_keys = np.array(list(da.attrs["stress_repeats"].keys()))
            stress_repeats_values = np.array(list(da.attrs["stress_repeats"].values()))
            package_times, inds = np.unique(
                np.concatenate([ds_times, stress_repeats_keys]), return_index=True
            )
            # Times to write in the runfile
            runfile_times = np.concatenate([ds_times, stress_repeats_values])[inds]
            starts = timeutil.forcing_starts(package_times, globaltimes)
        elif "stress_periodic" in da.attrs:
            runfile_times = package_times = ds_times
            starts = [da.attrs["stress_periodic"][time] for time in ds_times]
        else:
            runfile_times = package_times = ds_times
            starts = timeutil.forcing_starts(package_times, globaltimes)

        return runfile_times, starts

    def compose(
        self,
        directory,
        globaltimes,
        nlayer,
        composition=None,
        system_index=1,
        compose_projectfile=True,
        pkggroup_time=None,
    ):
        """
        Composes all variables for one system.
        """

        if composition is None:
            composition = Vividict()

        for data_var in self._variable_order:
            self._compose_values_timelayer(
                data_var,
                globaltimes,
                directory,
                nlayer,
                values=composition,
                system_index=system_index,
                compose_projectfile=compose_projectfile,
                pkggroup_times=pkggroup_time,
            )

        return composition

    def _compose_values_timelayer(
        self,
        varname,
        globaltimes,
        directory,
        nlayer,
        values=None,
        system_index=1,
        compose_projectfile=True,
        pkggroup_times=None,
    ):
        """
        Composes paths to files, or gets the appropriate scalar value for a
        single variable in a dataset.

        Parameters
        ----------
        varname : str
            variable name of the DataArray
        globaltimes : list, np.array
            Holds the global times, i.e. the combined unique times of every
            boundary condition that are used to define the stress periods.  The
            times of the BoundaryCondition do not have to match all the global
            times. When a globaltime is not present in the BoundaryCondition,
            the value of the first previous available time is filled in. The
            effective result is a forward fill in time.
        directory : str
            Path to working directory, where files will be written.  Necessary
            to generate the paths for the runfile.
        nlayer : int
            Number of layers
        values : Vividict
            Vividict (tree-like dictionary) to which values should be added
        system_index : int
            System number. Defaults to 1, but for package groups it is used
        compose_projectfile : bool
            Compose values in a hierarchy suitable for the projectfile
        pkggroup_times : optional, list, np.array
            Holds the package_group times.  Packages in one group need to be
            synchronized for iMODFLOW.

        Returns
        -------
        values : Vividict
            A nested dictionary containing following the projectfile hierarchy:
            ``{_pkg_id : {stress_period : {varname : {system_index : {lay_nr : value}}}}}``
            or runfile hierarchy:
            ``{stress_period : {_pkg_id : {varname : {system_index : {lay_nr : value}}}}}``
            where 'value' can be a scalar or a path to a file.
            The stress period number may be the wildcard '?'.
        """

        if values == None:
            values = Vividict()

        da = self[varname]

        if ("time" in self.dataset.coords) or (pkggroup_times is not None):
            runfile_times, starts = self._get_runfile_times(
                da, globaltimes, ds_times=pkggroup_times
            )

            for time, start in zip(runfile_times, starts):
                if compose_projectfile == True:
                    values[self._pkg_id][start][varname][
                        system_index
                    ] = self._compose_values_layer(
                        varname, directory, nlayer, time=time
                    )
                else:  # render runfile
                    values[start][self._pkg_id][varname][
                        system_index
                    ] = self._compose_values_layer(
                        varname, directory, nlayer, time=time
                    )

        else:
            if compose_projectfile == True:
                values[self._pkg_id]["1"][varname][
                    system_index
                ] = self._compose_values_layer(varname, directory, nlayer, time=None)
            else:  # render runfile
                values["1"][self._pkg_id][varname][
                    system_index
                ] = self._compose_values_layer(varname, directory, nlayer, time=None)

        return values


class TopBoundaryCondition(BoundaryCondition, abc.ABC):
    """
    Abstract base class for boundary conditions that are only assigned to
    the first layer, namely the Recharge and EvapoTranspiration package.
    """

    _template_projectfile = jinja2.Template(
        # Specify amount of timesteps for a package
        # 1 indicates if package is active or not
        '{{"{:04d}".format(package_data|length)}}, ({{pkg_id}}), 1, {{name}}, {{variable_order}}\n'
        "{%- for time_key, time_data in package_data.items()%}\n"
        # Specify stress period
        # Specify amount of variables and entries(nlay, nsys) to be expected
        "{{times[time_key]}}\n"
        '{{"{:03d}".format(time_data|length)}}, {{"{:03d}".format(n_entry)}}\n'
        "{%-    for variable in variable_order%}\n"  # Preserve variable order
        "{%-        for system, system_data in time_data[variable].items() %}\n"
        # Recharge only applied to first layer
        "{%-            set value = system_data[1]%}\n"
        "{%-            if value is string %}\n"
        # If string then assume path:
        # 1 indicates the layer is activated
        # 2 indicates the second element of the final two elements should be read
        # 001 indicates recharge is applied to the first layer
        # 1.000 is the multiplication factor
        # 0.000 is the addition factor
        # -9999 indicates there is no data, following iMOD usual practice
        "1, 2, 001, 1.000, 0.000, -9999., {{value}}\n"
        "{%-            else %}\n"
        # Else assume a constant value is provided
        '1, 1, 001, 1.000, 0.000, {{value}}, ""\n'
        "{%-            endif %}\n"
        "{%-        endfor %}\n"
        "{%-    endfor %}\n"
        "{%- endfor %}\n"
    )

    def _select_first_layer_composition(self, composition):
        """Select first layer in an exisiting composition."""
        composition_first_layer = Vividict()

        # Loop over nested dict, it is not pretty
        for (a, aa) in composition[self._pkg_id].items():
            for (b, bb) in aa.items():
                for (c, cc) in bb.items():
                    composition_first_layer[a][b][c][1] = cc[1]
        return composition_first_layer

    def compose(
        self,
        directory,
        globaltimes,
        nlayer,
        composition=None,
        compose_projectfile=True,
    ):

        composition = super(__class__, self).compose(
            directory,
            globaltimes,
            nlayer,
            composition,
            compose_projectfile=compose_projectfile,
        )

        composition[self._pkg_id] = self._select_first_layer_composition(composition)

        return composition
