import imod

import abc
import xarray as xr
import numpy as np

import jinja2
import pathlib

from imod import util
from imod.wq import timeutil


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
    Base package for the different iMODFLOW packages.
    Package is used to share methods for specific packages with no time
    component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    Every package contains a ``_pkg_id`` for identification.
    Used to check for duplicate entries, or to group multiple systems together
    (riv, ghb, drn).

    The ``_template_runfile`` attribute is the template for a section of the runfile.
    This is filled in based on the metadata from the DataArrays that are within
    the Package. Same applies to ``_template_projectfile`` for the projectfile.
    """

    __slots__ = ("_pkg_id", "_variable_order")

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

        for varname in self.dataset.data_vars:
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

            starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)
            for itime, start_end in enumerate(starts_ends):
                # TODO: this now fails on a non-dim time too
                # solution roughly the same as for layer above?
                values[start_end] = da.isel(time=itime).values[()]
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
    BoundaryCondition is used to share methods for specific stress packages with a time component.

    It is not meant to be used directly, only to inherit from, to implement new packages.
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

    def _add_timemap(self, varname, value, use_cftime):
        if value is not None:
            if varname not in self:
                raise ValueError(
                    f"{varname} does not occur in {self}\n cannot add timemap"
                )
            if "time" not in self[varname].coords:
                raise ValueError(
                    f"{varname} in {self}\n does not have dimension time, cannot add timemap."
                )

            # Replace both key and value by the right datetime type
            d = {
                timeutil.to_datetime(k, use_cftime): timeutil.to_datetime(v, use_cftime)
                for k, v in value.items()
            }
            self[varname].attrs["timemap"] = d

    def _get_runfile_times(self, da, globaltimes, ds_times=None):
        if ds_times is None:
            ds_times = self.dataset.coords["time"].values

        if "timemap" in da.attrs:
            timemap_keys = np.array(list(da.attrs["timemap"].keys()))
            timemap_values = np.array(list(da.attrs["timemap"].values()))
            package_times, inds = np.unique(
                np.concatenate([ds_times, timemap_keys]), return_index=True
            )
            # Times to write in the runfile
            runfile_times = np.concatenate([ds_times, timemap_values])[inds]
        else:
            runfile_times = package_times = ds_times

        starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)

        return runfile_times, starts_ends

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
        nlayer : int
            Number of layers
        values : Vividict
            Vividict (tree-like dictionary) to which values should be added
        system_index : int
            System number. Defaults to 1, but for package groups it is used
        compose_projectfile : bool
            Compose values in a hierarchy suitable for the projectfile
        pkggroup_times : optional, list, np.array
            Holds the package_group times.
            Packages in one group need to be synchronized for iMODFLOW.

        Returns
        -------
        values : Vividict
            A nested dictionary containing following the projectfile hierarchy:
            {_pkg_id : {stress_period : {varname : {system_index : {lay_nr : value}}}}}
            or runfile hierarchy:
            {stress_period : {_pkg_id : {varname : {system_index : {lay_nr : value}}}}}
            where 'value' can be a scalar or a path to a file.
            The stress period number may be the wildcard '?'.
        """

        if values == None:
            values = Vividict()

        da = self[varname]

        if ("time" in self.dataset.coords) or (pkggroup_times is not None):
            runfile_times, starts_ends = self._get_runfile_times(
                da, globaltimes, ds_times=pkggroup_times
            )

            for time, start_end in zip(runfile_times, starts_ends):
                if compose_projectfile == True:
                    values[self._pkg_id][start_end][varname][
                        system_index
                    ] = self._compose_values_layer(
                        varname, directory, nlayer, time=time
                    )
                else:  # render runfile
                    values[start_end][self._pkg_id][varname][
                        system_index
                    ] = self._compose_values_layer(
                        varname, directory, nlayer, time=time
                    )

        else:
            if compose_projectfile == True:
                values[self._pkg_id]["steady-state"][varname][
                    system_index
                ] = self._compose_values_layer(varname, directory, nlayer, time=None)
            else:  # render runfile
                values["steady-state"][self._pkg_id][varname][
                    system_index
                ] = self._compose_values_layer(varname, directory, nlayer, time=None)

        return values
