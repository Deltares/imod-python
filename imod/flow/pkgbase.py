import abc
import pathlib

import cftime
import jinja2
import numpy as np
import xarray as xr

import imod
from imod.flow import timeutil
from imod.util.nested_dict import initialize_nested_dict, set_nested


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

    The ``_template_projectfile`` attribute is the template for a section of the
    projectfile.  This is filled in based on the metadata from the DataArrays that
    are within the Package.
    """

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
    def from_file(cls, path, **kwargs):
        """
        Loads an imod-flow package from a file (currently only netcdf and zarr are supported).
        Note that it is expected that this file was saved with imod.flow.Package.save(),
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
        package : imod.flow.Package
            Returns a package with data loaded from file.

        Examples
        --------

        To load a package from a file, e.g. a River package:

        >>> river = imod.flow.River.from_file("river.nc")

        For large datasets, you likely want to process it in chunks. You can
        forward keyword arguments to ``xarray.open_dataset()`` or
        ``xarray.open_zarr()``:

        >>> river = imod.flow.River.from_file("river.nc", chunks={"time": 1})

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

    def compose(self, directory, globaltimes, nlayer, composition=None, **ignored):
        """
        Composes package, not useful for boundary conditions

        Parameters
        ----------
        directory : str
            Path to working directory, where files will be written.
            Necessary to generate the paths for the projectfile.
        globaltimes : list #TODO make this an *arg, change order.
            Not used, only included to comply with BoundaryCondition.compose
        nlayer : int
            Number of layers
        **ignored
            Contains keyword arguments unused for packages
        """

        composition = initialize_nested_dict(3)

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
        return str(imod.util.path.compose(d, pattern).resolve())

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
            Necessary to generate the paths for the projectfile.
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

        values = initialize_nested_dict(1)
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
                    dim_sel = {}
                    if "layer" in da.dims:
                        dim_sel["layer"] = layer
                    if "time" in da.dims:
                        dim_sel["time"] = time
                    values[layer] = da.sel(**dim_sel).values[()]

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
                imod.util.time.to_datetime_internal(
                    k, use_cftime
                ): imod.util.time.to_datetime_internal(v, use_cftime)
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
        d = {
            imod.util.time.to_datetime_internal(k, use_cftime): v
            for k, v in periods.items()
        }

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
        system_index=1,
        pkggroup_time=None,
    ):
        """
        Composes all variables for one system.

        Parameters
        ----------
        globaltimes : list, np.array
            Holds the global times, i.e. the combined unique times of every
            boundary condition that are used to define the stress periods.  The
            times of the BoundaryCondition do not have to match all the global
            times. When a globaltime is not present in the BoundaryCondition,
            the value of the first previous available time is filled in. The
            effective result is a forward fill in time.
        directory : str
            Path to working directory, where files will be written.  Necessary
            to generate the paths for the projectfile.
        nlayer : int
            Number of layers
        system_index : int
            System number. Defaults to 1, but for package groups it is used
        pkggroup_times : optional, list, np.array
            Holds the package_group times.  Packages in one group need to be
            synchronized for iMODFLOW.

        Returns
        -------
        composition : defaultdict
            A nested dictionary containing following the projectfile hierarchy:
            ``{_pkg_id : {stress_period : {varname : {system_index : {lay_nr : value}}}}}``
            where 'value' can be a scalar or a path to a file.
            The stress period number may be the wildcard '?'.
        """

        composition = initialize_nested_dict(5)

        for data_var in self._variable_order:
            keys_ls, values = self._compose_values_timelayer(
                data_var,
                globaltimes,
                directory,
                nlayer,
                system_index=system_index,
                pkggroup_times=pkggroup_time,
            )
            for keys, value in zip(keys_ls, values):
                set_nested(composition, keys, value)

        return composition

    def _compose_values_timelayer(
        self,
        varname,
        globaltimes,
        directory,
        nlayer,
        system_index=1,
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
            to generate the paths for the projectfile.
        nlayer : int
            Number of layers
        system_index : int
            System number. Defaults to 1, but for package groups it is used
        pkggroup_times : optional, list, np.array
            Holds the package_group times.  Packages in one group need to be
            synchronized for iMODFLOW.

        Returns
        -------
        keys : list of lists
            Contains keys for nested dict in the right order
        values : list
            List with composed layers

        """

        values = []
        keys = []

        da = self[varname]

        # Check if time defined for one variable in package
        # iMODFLOW's projectfile requires all variables for every timestep
        if self._hastime():
            compose_with_time = True
            times_for_path, starts = self._get_runfile_times(
                da, globaltimes, ds_times=pkggroup_times
            )
        # Catch case where the package has no time, but another
        # package in the group has. So the path has no time, but
        # needs a time entry in the composition
        elif pkggroup_times is not None:
            compose_with_time = True
            _, starts = self._get_runfile_times(
                da, globaltimes, ds_times=pkggroup_times
            )
            times_for_path = [None] * len(starts)
        else:
            compose_with_time = False

        if compose_with_time:
            for time, start in zip(times_for_path, starts):
                composed_layers = self._compose_values_layer(
                    varname, directory, nlayer, time=time
                )
                values.append(composed_layers)
                keys.append([self._pkg_id, start, varname, system_index])

        else:
            composed_layers = self._compose_values_layer(
                varname, directory, nlayer, time=None
            )
            values.append(composed_layers)
            keys.append([self._pkg_id, "steady-state", varname, system_index])

        return keys, values


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
        composition_first_layer = initialize_nested_dict(5)

        # Loop over nested dict, it is not pretty
        for a, aa in composition[self._pkg_id].items():
            for b, bb in aa.items():
                for c, cc in bb.items():
                    composition_first_layer[a][b][c][1] = cc[1]
        return composition_first_layer

    def compose(
        self,
        directory,
        globaltimes,
        nlayer,
    ):
        composition = super().compose(
            directory,
            globaltimes,
            nlayer,
        )

        composition[self._pkg_id] = self._select_first_layer_composition(composition)

        return composition
