import imod

import abc
import xarray as xr
import numpy as np

import jinja2
import pathlib

from imod import util
from imod.wq import timeutil

from imod.flow.util import Vividict

class Package(abc.ABC): #TODO: Abstract base class really necessary? Are we using abstract methods?
    """   
    Base package for the different iMODFLOW packages.
    Package is used to share methods for specific packages with no time
    component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    Every package contains a ``_pkg_id`` for identification.
    Used to check for duplicate entries, or to group multiple systems together
    (riv, ghb, drn).

    The ``_template`` attribute is the template for a section of the runfile.
    This is filled in based on the metadata from the DataArrays that are within
    the Package.

    """

    __slots__ = ("_pkg_id", "_variable_order")
    
    _template = jinja2.Template(
        "{%- for layer, path in variable_data %}\n"
        '{{layer}}, 1.0, 0.0, "{{path}}"\n'
        "{%- endfor %}\n"
    )

    _template_projectfile = jinja2.Template(
        "0001, ({{pkg_id}}), name, {{variable_order}}\n"
        '001, {{"{:03d}".format(nsub)}}\n'
        "{%- for variable in variable_order%}\n" #Preserve variable order
        "{%-    for layer, value in package_data[variable].items()%}\n"
        "{%-        if value is string %}\n" #If string then assume path
        '1, 2, {{"{:03d}".format(layer)}}, 1.000, 0.000, -9999., {{value}}\n'
        "{%-        else %}\n"
        '1, 1, {{"{:03d}".format(layer)}}, 1.000, 0.000, {{value}}, ""\n'
        "{%-        endif %}\n"
        "{%-    endfor %}\n"
        "{%- endfor %}\n"
    )
    
    def __init__(self):
        super(__class__, self).__init__()
        self.dataset = xr.Dataset()

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    #TODO:
    #def __getattribute__(self, name):
    #"implement the: https://github.com/xgcm/xgcm/issues/225#issuecomment-762248339"
    #    pass

    def compose(
        self, directory, globaltimes, nlayer, 
        compose_projectfile=True, composition=None
    ):
        """
        Composes package, not useful for boundary conditions

        Parameters
        ----------
        directory : str
            Path to working directory, where files will be written.
            Necessary to generate the paths for the runfile.
        globaltimes : list #TODO make this an *arg, change order.
            Not used, only included to comply with BoundaryCondition.compose
        nlayer : int
            Number of layers.
        """

        if composition is None:
            composition = Vividict()

        for varname in self.dataset.data_vars:
            composition[self._pkg_id][varname] = self._compose_values_layer(varname, directory, nlayer)

        return composition

    def _compose_path(self, d, pattern=None):
        # d : dict
        # pattern : string or re.pattern
        return util.compose(d, pattern).as_posix()

    def _compose_values_layer(
        self, varname, directory, nlayer, time=None, da=None
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

        Returns
        -------
        values : dict
            Dictionary containing the {layer number: path to file}.
            Alternatively: {layer number: scalar value}. The layer number may be
            a wildcard (e.g. '?').
        """
        pattern = "{name}"

        values = Vividict()
        if da is None:
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
                values["?"] = self._compose_path(d, pattern=pattern)
            else:
                values["?"] = da.values[()]

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
            values["?"] = da.values[()]

        return values

    def _render(self, render_projectfile=True, **kwargs):
        """
        Rendering method for simple keyword packages (vdf, pcg).

        Returns
        -------
        rendered : str
            The rendered runfile part for a single boundary condition system.
        """
        if render_projectfile:
            return self._template_projectfile.format(**kwargs)
        else:
            return self._template.format(**kwargs)

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

    #PSEUDO template for projectfile:
    #{ntimesteps}, (_pkg_id),1, name, [list_variables]
    #{timestamp}
    #{nvar}, {nlay * nsys}
    #1,{file_flag}, {{:03d}.format(lay_nr)}, 1.0000, 0.0000, {constant_value}, {path}
    
    #if file_flag == 1 for constant_value used, if file_flag == 2 path used

    #PSEUDO template for runfile:
    #{timestep_nr},1.0,{timestamp},-1
    #{nlay * nsys}, ({_pkg_id})
    #{lay_nr}, 1.0, 0.0, {path}

    #EXAMPLE projectfile
    #0001,(DRN),1, Drainage,[CON,DEL]
    #2002-01-01 00:00:00
    #002,001
    #1,2, 001,   1.000000    ,   0.000000    ,  -999.9900    ,'z:\buisdrainage_conductance.idf'
    #1,2, 001,   1.000000    ,   0.000000    ,  -999.9900    ,'z:\buisdrainage_peil.idf'

    #EXAMPLE runfile
    #1,1.0,19710101000000,-1
    #6, (ghb)
    #1, 1.0, 0.0, "c:\ghb-cond_19710101000000_l1.idf"
    #2, 1.0, 0.0, "c:\ghb-cond_19710101000000_l2.idf"
    #3, 1.0, 0.0, "c:\ghb-cond_19710101000000_l3.idf"
    #1, 1.0, 0.0, "c:\ghb-cond-sys2_19710101000000_l1.idf"
    #2, 1.0, 0.0, "c:\ghb-cond-sys2_19710101000000_l2.idf"
    #3, 1.0, 0.0, "c:\ghb-cond-sys2_19710101000000_l3.idf"
    #1, 1.0, 0.0, "c:\ghb-head_19710101000000_l1.idf"
    #2, 1.0, 0.0, "c:\ghb-head_19710101000000_l2.idf"
    #3, 1.0, 0.0, "c:\ghb-head_19710101000000_l3.idf"
    #1, 1.0, 0.0, "c:\ghb-head-sys2_19710101000000_l1.idf"
    #2, 1.0, 0.0, "c:\ghb-head-sys2_19710101000000_l2.idf"
    #3, 1.0, 0.0, "c:\ghb-head-sys2_19710101000000_l3.idf"

    _template_projectfile = jinja2.Template(
        '{{"{:03d}".format(package_data|length)}}, ({{pkg_id}}), name, {{variable_order}}\n'
        "{%- for time_key, time_data in package_data.items()%}\n"
        '{{times[time_key]}}\n'
        '{{"{:03d}".format(time_data|length)}}, {{"{:03d}".format(nsub)}}\n'
        "{%-    for variable in variable_order%}\n" #Preserve variable order
        "{%-        for system, system_data in time_data[variable].items() %}\n"
        "{%-            for layer, value in system_data.items() %}\n"
        "{%-                if value is string %}\n" #If string then assume path
        '1, 2, {{"{:03d}".format(layer)}}, 1.000, 0.000, -9999., {{value}}\n'
        "{%-                else %}\n"
        '1, 1, {{"{:03d}".format(layer)}}, 1.000, 0.000, {{value}}, ""\n'
        "{%-                endif %}\n"
        "{%-            endfor %}\n"
        "{%-        endfor %}\n"
        "{%-    endfor %}\n"
        "{%- endfor %}\n"
    )

    def _get_runfile_times(self, da, globaltimes):
            da_times = da.coords["time"].values
            if "timemap" in da.attrs:
                timemap_keys = np.array(list(da.attrs["timemap"].keys()))
                timemap_values = np.array(list(da.attrs["timemap"].values()))
                package_times, inds = np.unique(
                    np.concatenate([da_times, timemap_keys]), return_index=True
                )
                # Times to write in the runfile
                runfile_times = np.concatenate([da_times, timemap_values])[inds]
            else:
                runfile_times = package_times = da_times

            starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)

            return runfile_times, starts_ends

    def compose(self, directory, globaltimes, nlayer, 
        composition=None, sys_nr=1, compose_projectfile=True):
        """
        Composes all variables for one system. 
        """
        
        if composition is None:
            composition = Vividict()

        for data_var in self.dataset.data_vars:
            self._compose_values_timelayer(
                data_var, globaltimes, directory, nlayer, 
                values = composition, sys_nr = sys_nr,
                compose_projectfile=compose_projectfile)
        
        return composition

    def _compose_values_timelayer(
        self, varname, globaltimes, directory, nlayer, 
        values = None, sys_nr=1, da=None,
        compose_projectfile=True
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
        values : Vividict
            Vividict (tree-like dictionary) to which values should be added
        sys_nr : int
            System number. Defaults as 1, but for package groups it 
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

        if values == None:
            values = Vividict()

        if da is None:
            da = self[varname]

        args = (varname, directory)
        kwargs = dict(nlayer=nlayer, da=da, time=None)

        if "time" in da.coords:
            runfile_times, starts_ends = self._get_runfile_times(da, globaltimes)

            for time, start_end in zip(runfile_times, starts_ends):
                kwargs["time"] = time
                if compose_projectfile == True:
                    values[self._pkg_id][start_end][varname][sys_nr] = self._compose_values_layer(*args, **kwargs)
                    #values[self._pkg_id][start_end][varname][sys_nr] = self._compose_values_layer(*args, **kwargs)
                else: #render runfile
                    values[start_end][self._pkg_id][varname][sys_nr] = self._compose_values_layer(*args, **kwargs)

        else:
            if compose_projectfile == True:
                values[self._pkg_id]["steady-state"][varname][sys_nr] = self._compose_values_layer(*args, **kwargs)
            else:
                values["steady-state"][self._pkg_id][varname][sys_nr] = self._compose_values_layer(*args, **kwargs)

        return values 