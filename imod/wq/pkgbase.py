import pathlib

import jinja2
import numpy as np
import pandas as pd
import xarray as xr
import imod
from imod import util
from imod.wq import timeutil


class Package(xr.Dataset):
    """
    Base package for the different SEAWAT packages.
    Every package contains a `_pkg_id` for identification.
    Used to check for duplicate entries, or to group multiple systems together
    (riv, ghb, drn).

    The `_template` attribute is the template for a section of the runfile.
    This is filled in based on the metadata from the DataArrays that are within
    the Package.

    The `_keywords` attribute is a dictionary that's used to replace
    keyword argument by integer arguments for SEAWAT.
    """

    def _replace_keyword(self, d, key):
        """
        Method to replace a readable keyword value by the corresponding cryptic
        integer value that SEAWAT demands.

        Dict `d` is updated in place.

        Parameters
        ----------
        d : dict
            Updated in place.
        key : str
            key of value in dict `d` to replace.
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
        d = {k: v.values for k, v in self.data_vars.items()}
        if hasattr(self, "_keywords"):
            for key in self._keywords.keys():
                self._replace_keyword(d, key)
        return self._template.format(**d)

    def _compose_values_layer(self, varname, directory, time=None, da=None):
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
        d : dict, optional
            Partially filled in dictionary with the parts of the filename.
            Used for transient data.
        da : xr.DataArray, optional
            In some cases fetching the DataArray by varname is not desired.
            It can be passed directly via this optional argument.

        Returns
        -------
        values : dict
            Dictionary containing the {layer number: path to file}.
            Alternatively: {layer number: scalar value}. The layer number may be
            a wildcard (e.g. '?').
        """
        values = {}
        if da is None:
            da = self[varname]

        d = {"directory": directory, "name": varname, "extension": ".idf"}
        if time is not None:
            d["time"] = time

        # Scalar value or not?
        # If it's a scalar value we can immediately write
        # otherwise, we have to write a file path
        if "y" not in da.coords and "x" not in da.coords:
            idf = False
        else:
            idf = True

        if "layer" not in da.coords:
            if idf:
                values["?"] = util.compose(d)
            else:
                values["?"] = da.values[()]

        else:
            for layer in np.atleast_1d(da.coords["layer"].values):
                if idf:
                    d["layer"] = layer
                    values[layer] = util.compose(d)
                else:
                    if "layer" in da.dims:
                        values[layer] = da.sel(layer=layer).values[()]
                    else:
                        values[layer] = da.values[()]

        return values

    def _compose_values_time(self, varname, globaltimes):
        da = self[varname]
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

        return values

    def save(self, directory):
        for name, da in self.data_vars.items():
            if "y" in da.coords and "x" in da.coords:
                path = pathlib.Path(directory).joinpath(name)
                imod.idf.save(path, da)


class BoundaryCondition(Package):
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

    def _compose_values_timelayer(self, varname, globaltimes, directory, da=None):
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
            da = self[varname]

        if "time" in da.coords:
            package_times = da.coords["time"].values

            starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)

            for time, start_end in zip(package_times, starts_ends):
                values[start_end] = self._compose_values_layer(varname, directory, time)

        else:
            values["?"] = self._compose_values_layer(varname, directory)

        return values

    def _max_active_n(self, varname, nlayer):
        """
        Determine the maximum active number of cells that are active
        during a stress period.

        Parameters
        ----------
        varname : str
            name of the variable to use to calculate the maximum number of
            active cells. Generally conductance.
        nlayer : int
            number of layers, taken from ibound.
        """
        if "time" in self[varname].coords:
            nmax = int(self[varname].groupby("time").count().max())
        else:
            nmax = int(self[varname].count())
        if not "layer" in self.coords:  # Then it applies to every layer
            nmax *= nlayer
        return nmax
        # TODO: save this as attribute so it doesn't have to be recomputed for SSM?
        # Maybe call in __init__, then.
        # or just call it before render
        # and always make sure to call SSM render afterwards

    def _render(self, directory, globaltimes, system_index):
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
        mapping = tuple([(k, v) for k, v in self._mapping if v in self.data_vars])
        d = {"mapping": mapping, "system_index": system_index}
        dicts = {}

        for varname in self.data_vars.keys():
            dicts[varname] = self._compose_values_timelayer(
                varname, globaltimes, directory
            )

        d["dicts"] = dicts

        return self._template.render(d)

    def _render_ssm(self, directory, globaltimes):
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

        if "concentration" not in self:
            return ""

        d = {}
        d["pkg_id"] = self._pkg_id
        if "species" in self["concentration"].coords:
            concentration = {}
            for i, species in enumerate(self["concentration"]["species"].values):
                concentration[i + 1] = self._compose_values_timelayer(
                    varname="concentration",
                    da=self["concentration"].sel(species=species),
                    globaltimes=globaltimes,
                    directory=directory,
                )
        else:
            concentration = {
                1: self._compose_values_timelayer(
                    varname="concentration",
                    da=self["concentration"],
                    globaltimes=globaltimes,
                    directory=directory,
                )
            }
        d["concentration"] = concentration
        return self._ssm_template.render(d)
