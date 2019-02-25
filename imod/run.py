import os
import xarray as xr
import pandas as pd
import numpy as np
import itertools
import jinja2
from imod import util
from imod import idf
from pathlib import Path
from collections import OrderedDict

# change into namedtuple?
package_schema = OrderedDict(
    [
        ("bnd", {}),
        ("shd", {}),
        ("kdw", {}),
        ("vcw", {}),
        ("khv", {}),
        ("kva", {}),
        ("kvv", {}),
        ("sto", {}),
        ("ssc", {}),
        ("top", {}),
        ("bot", {}),
        ("pwt", {}),
        ("ani", {"order": ("angle", "factor")}),
        ("hfb", {"order": ("factor", "resistance")}),
    ]
)

stress_period_schema = OrderedDict(
    [
        ("wel", {"systems": True}),
        ("drn", {"systems": True, "order": ("cond", "bot")}),
        ("riv", {"systems": True, "order": ("cond", "stage", "bot", "inff")}),
        ("ghb", {"systems": True, "order": ("cond", "head")}),
        ("rch", {"systems": False}),
        ("chd", {"systems": False}),
    ]
)

default_runfile = OrderedDict(
    [
        ("modelname", "results"),
        ("sdate", 0),
        ("nscl", 1),
        ("iconchk", 0),
        ("iipf", 0),
        ("iarmwp", 0),
        ("nmult", 1),
        ("idebug", 0),
        ("iexport", 0),
        ("iposwel", 0),
        ("iscen", 0),
        ("outer", 150),
        ("inner", 30),
        ("hclose", 0.001),
        ("qclose", 100.0),
        ("relax", 0.98),
        ("buffer", 0.0),
        ("output", {}),
    ]
)

seawat_package_schema = OrderedDict(
    [
        ("bnd", {}),
        ("icbund", {}),
        ("top", {}),
        ("bot", {}),
        ("thickness", {}),
        ("shd", {}),
        ("sconc", {}),
        ("khv", {}),
        ("kva", {}),
        ("sto", {}),
        ("por", {}),
        ("dsp", {"order": ("al",)}),
    ]
)

seawat_period_schema = OrderedDict(
    [
        ("wel", {"systems": True, "order": ("rate", "conc")}),
        ("drn", {"systems": True, "order": ("cond", "bot", "conc")}),
        ("riv", {"systems": True, "order": ("cond", "stage", "bot", "dens", "conc")}),
        ("ghb", {"systems": True, "order": ("cond", "head", "dens", "conc")}),
        ("rch", {"systems": True, "order": ("rate", "conc")}),
        ("chd", {"systems": True, "order": ("head", "conc")}),
    ]
)

seawat_default_runfile = OrderedDict(
    [
        # gen
        ("modelname", "results"),
        ("writehelp", True),
        # dis
        ("nstp", 1),
        ("sstr", "tr"),
        ("laycbd", 0),
        # bas6
        ("hnoflo", -9999.0),
        # oc
        ("savehead", True),
        ("saveconclayer", True),
        ("savebudget", False),
        ("saveheadtec", False),
        ("saveconctec", False),
        ("savevxtec", False),
        ("savevytec", False),
        ("savevztec", False),
        # lfp
        ("ilpfcb", 1),
        ("hdry", 1.0e30),
        ("nplpf", 0),
        ("laytyp", 0),
        ("layavg", 0),
        ("chani", 1.0),
        ("layvka", 0),
        # pcg
        ("mxiter", 100),
        ("iter1", 30),
        ("hclose", 0.0001),
        ("rclose", 1.0),
        ("relax", 0.98),
        ("nbpol", 0),
        ("iprpcg", 1),
        ("mutpcg", 1),
        # pksf
        ("pksf", False),
        ("mxiterpks", 1000),
        ("inneritpks", 30),
        ("hclosepks", 0.0001),
        ("rclosepks", 1.0),
        ("npc", 2),
        ("partopt", 0),
        ("pressakey", False),
        # btn
        ("cinact", -9999.0),
        ("thkmin", 0.01),
        ("nprs", 0),
        ("ifmtcn", -1),
        ("chkmas", True),
        ("nprmas", 10),
        ("nprobs", 1),
        ("tsmult", 1.0),
        ("dt0", 0.0),
        ("mxstrn", 10000.0),
        ("ttsmult", 1.0),
        ("ttsmax", 0.0),
        # adv
        ("mixelm", -1),
        ("percel", 1.0),
        ("mxpart", 100_000),
        ("itrack", 1),
        ("wd", 0.5),
        ("dceps", 0.0001),
        ("nplane", 2),
        ("npl", 0),
        ("nph", 8),
        ("npmin", 0),
        ("npmax", 16),
        ("interp", 1),
        ("nlsink", 2),
        ("npsink", 8),
        ("dchmoc", 0.001),
        # dsp
        ("trpt", 1.0),
        ("trpv", 1.0),
        ("dmcoef", 0.0001),
        # gcg
        ("mt3d_mxiter", 1000),
        ("mt3d_iter1", 300),
        ("mt3d_isolve", 2),
        # vdf
        ("mtdnconc", 1),
        ("mfnadvfd", 2),
        ("nswtcpl", 1),
        ("iwtable", 0),
        ("densemin", 1000.0),
        ("densemax", 1025.0),
        ("denseref", 1000.0),
        ("denseslp", 0.7143),
        # drn
        ("mxactd", 1.0e6),
        ("idrncb", 0),
        # chd
        ("mxactc", 1.0e6),
        # gbh
        ("mxactb", 1.0e6),
        ("ighbcb", 0),
        # riv
        ("mxactr", 1.0e6),
        ("irivcb", 0),
        # rch
        ("nrchop", 3),
        ("irchcb", 0),
        # wel
        ("mxactw", 1.0e6),
        ("iwelcb", 0),
        # ssm
        ("mxss", 1.0e6),
    ]
)


def _check_input(model, seawat=False):
    """
    Tests whether model and content is of appropriate type, generates new 
    OrderedDict to avoid destroying the model dict, and lowers keys for further 
    processing.

    Parameters
    ----------
    model : OrderedDict
        The OrderedDict containing the model data.
    seawat : bool
        Set True if model is seawat model
    
    Returns
    -------
    consumed_model: OrderedDict
        Copied `model` dict with lower case keys, which will be consumed during
        writing.
    """
    # TODO: needs a better name?

    assert isinstance(model, OrderedDict), "model must be an OrderedDict."
    consumed_model = OrderedDict()

    for key, value in model.items():
        if key.split("-")[0].lower() == "wel" and not seawat:
            assert isinstance(
                value, pd.DataFrame
            ), "wel package must be a pandas dataframe."
        elif seawat and key.lower() == "wel-rate":
            assert isinstance(
                value, pd.DataFrame
            ), "wel-rate must be a pandas dataframe."
        else:
            assert isinstance(
                value, xr.DataArray
            ), "{} must be an xarray DataArray.".format(key)

        consumed_model[key.lower()] = value

    return consumed_model


def _data_bounds(model, seawat=False):
    """ 
    Collects spatial bounds from bnd (bnd is therefore required):

    * nlay
    * cellsize
    * xmin
    * max
    * ymin
    * ymax
    and checks for consistency.

    Time bounds:

    * nper
    If applicable:
    * times
    * sdate
    * edate

    Parameters
    ----------
    model : OrderedDict
        The OrderedDict containing the model data.
    seawat : boolean
        Set to `True` if data is for an iMODSEAWAT model.

    """
    # TODO: Think of what should be done when ibound_l1 is not the largest
    # of the ibound layers...
    # and what should be used as pointer grid

    layers = [int(layer) for layer in np.atleast_1d(model["bnd"].coords["layer"])]
    # check if consecutive
    assert sorted(layers) == list(
        range(1, len(layers) + 1)
    ), "bnd layers must start at 1 and be consecutive."
    # Should we support multiple cellsizes within one model? Since imod just resamples
    dx, bnd_xmin, bnd_xmax, dy, bnd_ymin, bnd_ymax = util.spatial_reference(
        model["bnd"]
    )

    times = set()  # use a set to only allow unique values
    tvals = None
    for key, data in model.items():
        # assumes wel dataframe is in "tidy" or "long" format as defined in:
        # http://vita.had.co.nz/papers/tidy-data.pdf
        if isinstance(data, pd.DataFrame):
            if "time" in data.columns:
                # TODO: assert time is cftime
                tvals = data["time"].values
            else:
                tvals = None
            xmin = float(data["x"].min())
            xmax = float(data["x"].max())
            ymin = float(data["y"].min())
            ymax = float(data["y"].max())
            check_layers = pd.unique(data["layer"].values)

        else:  # then it should be a DataArray
            if "time" in data.coords:
                # TODO: assert time is cftime
                tvals = data["time"].values
            else:
                tvals = None
            _, xmin, xmax, _, ymin, ymax = util.spatial_reference(data)
            check_layers = np.atleast_1d(data["layer"].values)

        for layer in check_layers:
            # e.g. for recharge to top most layer, -1 is supported value
            if int(layer) == -1 and not seawat:
                pass
            else:
                assert (
                    int(layer) in layers  # should fail on NaNs, 0, negative
                ), "{} : layer {} falls outside of bnd".format(key, layer)

        assert xmin >= bnd_xmin, "{}: xmin falls outside of bnd.".format(key)
        assert xmax <= bnd_xmax, "{}: xmax falls outside of bnd.".format(key)
        assert ymin >= bnd_ymin, "{}: ymin falls outside of bnd.".format(key)
        assert ymax <= bnd_ymax, "{}: ymax falls outside of bnd.".format(key)

        if tvals is not None:
            try:
                times.update(tvals)
            except TypeError:  # if not an iterable
                times.add(tvals)

    d = OrderedDict()
    d["nrow"] = model["bnd"].y.size
    d["ncol"] = model["bnd"].x.size
    d["nlay"] = len(layers)
    d["dx"] = abs(dx)
    d["dy"] = abs(dy)
    d["xmin"] = bnd_xmin
    d["xmax"] = bnd_xmax
    d["ymin"] = bnd_ymin
    d["ymax"] = bnd_ymax

    if len(times) > 0:
        d["nper"] = len(times) - 1
        d["times"] = sorted(times)
        # TODO: cftime
    else:
        d["nper"] = 1

    return d


def _parse(key, stress_period_schema):
    """
    Parsing keys of stress period DataArrays.

    In naming:

    * Package is always required
    * Parameter is required (stage, cond, etc.) for some packages
    * System is optional for some packages

    This function first gets the name of the package, e.g. "riv", then if the 
    package has multiple fields, it checks if the field is valid, e.g. "stage".
    Finally, if the package supports multiple systems, it attempts to get the 
    system name.

    Package and field names are standardized; system names may be arbitrary (no dashes).

    Parameters
    ----------
    key : str
    stress_period_schema : OrderedDict
        Schema against which to validate key
    
    Returns
    -------
    dict
        dict containing package name, field, system
    """
    d = {}
    parts = key.split("-")
    name = parts.pop(0)
    d["name"] = name

    order = stress_period_schema[name].get("order", False)
    if order:
        try:
            field = parts.pop(0)
        except IndexError as e:
            raise ValueError(
                f"A field is missing for {key}. Required fields are: {','.join(order)}"
            ) from e
        assert (
            field in order
        ), "{} is not a field of {}. Possible values are: {}".format(
            field, name, ",".join(order)
        )
        d["field"] = field

    if stress_period_schema[name]["systems"]:
        try:
            d["system"] = parts.pop(0)
        except IndexError:
            pass
    else:  # if systems are not supported, parts list should be empty by now.
        assert len(parts) == 0, "{} does not support systems.".format(name)

    assert len(parts) == 0, "Parts: {} cannot be parsed as part of {} package.".format(
        str(parts), name
    )

    return d


def _groupby_field(package, stress_period_schema):
    """
    Groups imodflow package content by field (using `itertools.groupby`).

    E.g. riv-stage-sys1 and riv-stage-sys2 end up in the same "stage" group.

    Parameters
    ----------
    package : dict

    stress_period_schema: OrderedDict
        Schema against which to validate 

    Returns
    -------
    grouped : itertools.groupby
        Package contents grouped by field
    """
    list_package = []
    for key, value in package.items():
        list_package.append(
            {"key": key, "data": value, **_parse(key, stress_period_schema)}
        )
    # has to be sorted for groupby
    list_package = sorted(list_package, key=lambda x: x.get("field", "value"))
    # groupby fields
    grouped = itertools.groupby(list_package, key=lambda x: x.get("field", "value"))
    return grouped


def _sortby_field(package_data, name, stress_period_schema):
    """
    Sorts modflow package content according to schema.

    Parameters
    ----------
    package_data : dict
        data for a single imodflow package
    name : str
        name of the package
    stress_period_schema : OrderedDict

    Returns
    -------
    package_data : dict
        data for a single imodflow package, reordered for writing to 
        runfile

    """
    order = stress_period_schema[name].get("order", False)

    if order:
        sorted_data = OrderedDict()
        # will also check if names are okay, otherwise raises KeyError
        for field in order:
            try:
                sorted_data[field] = package_data[field]
            except KeyError as e:
                raise KeyError("Package {}: {} is missing.".format(name, field)) from e
        package_data = sorted_data
    return package_data


def _time_discretisation(times):
    """
    Generates dictionary containing stress period time discretisation data.

    Parameters
    ----------
    times : np.array
        Array containing containing time in a datetime-like format
    
    Returns
    -------
    OrderedDict
        OrderedDict with dates as strings for keys,
        stress period duration (in days) as values.
    """
    d = OrderedDict()
    for start, end in zip(times[:-1], times[1:]):
        # TODO: force cftime
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        period_name = start.strftime("%Y%m%d%H%M%S")
        timedelta = end - start
        duration = timedelta.days + timedelta.seconds / 86400.0
        d[period_name] = duration
    return d


def _pop_package(model, name):
    """Pops all entries for a single package from a model dict."""
    package = OrderedDict()
    # select package content on basis of package name
    package_keys = [key for key in model.keys() if key.split("-")[0] == name]
    for key in package_keys:
        package[key] = model.pop(key)
    return model, package


def _get_wel(item, times, directory):
    """
    Generates all the paths to IPFs for a single system of a package.

    Steady-state is assumed when `times` evaluates to `False`.

    The nesting of an item is in the following order: system - layer.
    Time is handled by use of associated ipf files (.txt).

    Parameters
    ----------
    item : OrderedDict
    times : np.array
        Array containing "global" times: the datetimes collected from all 
        model data objects. The times in the item are a subset.
    directory : str
        Directory in which the IDF/IPFs will be written

    Returns
    -------
    single_system : OrderedDict
        Dictionary containing the generated paths (IPF) for a single system. 
    """
    # When the time resolution of an associated .txt file is higher than the
    # time resolution of the stress periods, iMODFLOW (apparently) simply
    # takes the first value that falls within the stress period.
    # Because of the _data_bounds() function, we do not follow that approach
    # here: if the well data is high resolution, so will the stress periods.

    if not times:  # steady state case
        times = [0]

    df = item["data"]
    key = item["key"]

    single_system = OrderedDict()
    for layer in np.unique(df["layer"].values):
        layer = int(layer)
        periods = []
        d = {"directory": directory, "name": key, "layer": layer, "extension": ".ipf"}
        for _ in times:
            periods.append(util.compose(d))
        single_system[layer] = periods

    return single_system


def _get_system(item, times, directory):
    """
    Generates all the paths to IDFs for a single system of a package.

    Steady-state is assumed when `times` evaluates to `False`.

    The nesting of an item is in the following order: system - layer - time.

    Parameters
    ----------
    item : OrderedDict
    times : np.array
        Array containing "global" times: the datetimes collected from all 
        model data objects. The times in the item are a subset.
    directory : str
        Directory in which the IDF/IPFs will be written

    Returns
    -------
    single_system : OrderedDict
        Dictionary containing the generated paths (IDF) for a single system. 
    """
    # TODO: Currently all packages, except for WEL, have to be defined in the
    # first stress period (and when no time is assigned, it is assumed constant
    # for all stress periods); because a forward fill occurs for the periods.
    # What does iMODFLOW/iMODSEAWAT support?
    # Is it allowed to add a package in a later stress period?
    # Is it allowed to add an additional system in a later stress period?
    # (E.g. an ephemeral stream)
    # Is this a common/reasonable use case? What should we support?

    if not times:  # steady state case
        times = [0]

    da = item["data"]
    key = item["key"]

    if "time" in da.coords:
        # TODO: fix when cftime is default xarray
        package_times = [
            pd.to_datetime(t) for t in np.atleast_1d(da.coords["time"].values)
        ]

    single_system = OrderedDict()
    for layer in np.atleast_1d(da.coords["layer"].values):
        layer = int(layer)
        periods = []
        d = {"directory": directory, "name": key, "layer": layer, "extension": ".idf"}
        for globaltime in times:
            if "time" in da.coords:
                # forward fill
                time = list(filter(lambda t: t <= globaltime, package_times))[-1]
                d["time"] = time
            periods.append(util.compose(d))
        single_system[layer] = periods

    return single_system


def _get_period(package, times, directory, stress_period_schema):
    """ 
    Generates paths for all fields, systems, layers, and times of a package, 
    that has stress periods. 

    Parameters
    ----------
    package : OrderedDict
        Data for a single imodflow package
    times : np.array
        Array containing "global" times: the datetimes collected from all 
        model data objects. The times in the item are a subset.
    directory : str
        Directory in which the model will be written.
    stress_period_schema : OrderedDict
        Schema against which to validate package data.

    Returns
    -------
    package_data : OrderedDict
        Dictionary containing the generated paths (IDF, IPF) for a single package. 
    """
    name = list(package.keys())[0].split("-")[0]
    package_data = OrderedDict()

    grouped = _groupby_field(package, stress_period_schema)
    for field, group in grouped:
        systemdata = OrderedDict()
        for item in group:
            systemname = item.get("system", "default_system")
            if name == "wel" and isinstance(item["data"], pd.DataFrame):
                systemdata[systemname] = _get_wel(item, times, directory)
            else:
                systemdata[systemname] = _get_system(item, times, directory)
        package_data[field] = systemdata

    package_data = _sortby_field(package_data, name, stress_period_schema)
    return package_data


def _get_package(package, directory, package_schema):
    """ 
    Generates paths for all fields and layers of a package, that does not have
    stress periods.

    Parameters
    ----------
    package : OrderedDict
        Data for a single imodflow package
    directory : str
        Directory in which the model will be written.
    package_schema : OrderedDict
        Schema against which to validate package data.

    Returns
    -------
    package_data : OrderedDict
        Dictionary containing the generated paths (IDF, IPF) for a single package. 
    """
    package_data = OrderedDict()
    for key, da in package.items():
        name = key.split("-")[0]
        try:
            order = package_schema[name].get("order", ["value"])
        except KeyError as e:
            raise KeyError("Package {}.".format(name)) from e

    for field in order:
        single_field = {}
        # select field
        if field == "value":
            da = package[name]
        else:
            da = package["-".join([name, field])]

        for layer in np.atleast_1d(da.coords["layer"].values):
            layer = int(layer)
            d = {
                "directory": directory,
                "name": key,
                "layer": layer,
                "extension": ".idf",
            }
            single_field[layer] = util.compose(d)
        package_data[field] = single_field

    return package_data


def get_runfile(model, directory):
    """
    Generates an OrderedDict containing the values to be filled in in a runfile 
    template, from the data contained in `model`.
    These values are mainly the paths of the IDFs and IPFs, nested in such a 
    way that it can be easily unpacked when filling in the runfiles; 
    plus a fairly large number of configuration values.

    For packages that do not have stress periods, the nesting is:
    package - field - layer
    
    For packages that have stress periods, the nesting is:
    package - field - system - layer - time

    **Note**: every `xarray.DataArray` containing the data must have layer coordinates specified;
    use `da.assign_coords(layer=...)`.

    Parameters
    ----------
    model: OrderedDict
        Dictionary containing the model data.
    directory : str 
        Directory in which the model will be written (and therefore necessary 
        for generating paths)

    Returns
    -------
    parameter_values : OrderedDict
        OrderedDictionary containing all the values necessary for filling in a 
        runfile. Nested in such a way that it can be easily unpacked in a 
        template.
    
    """
    consumed_model = _check_input(model)
    bounds = _data_bounds(model)
    bounds.pop("dy")
    bounds["cellsize"] = bounds.pop("dx")
    times = bounds.pop("times", False)

    if isinstance(directory, str):
        directory = Path(directory)
    directory = directory.absolute()  # as iMODFLOW supports only absolute paths
    runfile_parameters = default_runfile.copy()
    runfile_parameters.update(bounds)

    packages = OrderedDict()
    stress_periods = OrderedDict()

    package_names = {key.split("-")[0] for key in consumed_model.keys()}
    PACKAGE_CONTENT = tuple(package_schema)
    STRESS_PERIOD_CONTENT = tuple(stress_period_schema)
    for name in package_names:
        consumed_model, package = _pop_package(consumed_model, name)
        path = directory.joinpath(name)
        if name in PACKAGE_CONTENT:
            package_data = _get_package(package, path, package_schema)
            packages[name] = package_data
        elif name in STRESS_PERIOD_CONTENT:
            stress_period_data = _get_period(package, times, path, stress_period_schema)
            stress_periods[name] = stress_period_data
        else:
            raise RuntimeError(
                "Package {}: invalid name, or package is not supported.".format(name)
            )
    # check if entire model is consumed
    assert (
        len(consumed_model) == 0
    ), "Model could not be completely written due to keys:{} ".format(
        list(consumed_model.keys())
    )

    runfile_parameters["packages"] = packages
    runfile_parameters["stress_periods"] = stress_periods
    runfile_parameters["output"]["shd"] = list(model["bnd"].layer.values)

    if times:
        runfile_parameters["time_discretisation"] = _time_discretisation(times)
    else:
        runfile_parameters["time_discretisation"] = OrderedDict([("steady-state", 0)])

    return runfile_parameters


def _jinja2_template(fname):
    """Loads and returns a template from imod package files."""
    loader = jinja2.PackageLoader("imod", "templates")
    env = jinja2.Environment(loader=loader)
    return env.get_template(fname)


def write_runfile(path, runfile_parameters):
    """
    Writes an IMODFLOW runfile from metadata collected from model by
    `imod.run.get_runfile()`.

    Parameters
    ----------
    path : str
        Path to write runfile contents to.
    runfile_parameters : OrderedDict
        Dictionary used to fill in runfile.
    
    Returns
    -------
    None
    """
    template = _jinja2_template("runfile.j2")
    out = template.render(**runfile_parameters)

    with open(path, "w") as f:
        f.write(out)


def seawat_get_runfile(model, directory):
    """
    Generates an OrderedDict containing the values to be filled in in a runfile 
    template, from the data contained in `model`, specifically for an
    IMODSEAWAT model.

    These values are mainly the paths of the IDFs and IPFs, nested in such a 
    way that it can be easily unpacked when filling in the runfiles; 
    plus a fairly large number of configuration values.

    For packages that do not have stress periods, the nesting is:
    package - field - layer
    
    For packages that have stress periods, the nesting is:
    package - field - system - layer - time

    **Note**: every `xarray.DataArray` containing the data must have layer specified;
    use `da.assign_coords(layer=...)`.

    Parameters
    ----------
    model: OrderedDict
        Dictionary containing the model data.
    directory : str
        Directory in which the model will be written (and therefore necessary 
        for generating paths)

    Returns
    -------
    parameter_values : OrderedDict
        OrderedDictionary containing all the values necessary for filling in a 
        runfile. Nested in such a way that it can be easily unpacked in a 
        template.
    
    """
    consumed_model = _check_input(model, seawat=True)
    bounds = _data_bounds(model, seawat=True)
    times = bounds.pop("times", False)

    if isinstance(directory, str):
        directory = Path(directory)
    directory = directory.absolute()  # as iMODFLOW supports only absolute paths
    runfile_parameters = seawat_default_runfile.copy()
    runfile_parameters.update(bounds)

    packages = OrderedDict()
    stress_periods = OrderedDict()

    package_names = {key.split("-")[0] for key in consumed_model.keys()}
    PACKAGE_CONTENT = tuple(seawat_package_schema)
    STRESS_PERIOD_CONTENT = tuple(seawat_period_schema)
    for name in package_names:
        consumed_model, package = _pop_package(consumed_model, name)
        path = directory.joinpath(name)
        if name in PACKAGE_CONTENT:
            package_data = _get_package(package, path, seawat_package_schema)
            packages[name] = package_data
        elif name in STRESS_PERIOD_CONTENT:
            stress_period_data = _get_period(package, times, path, seawat_period_schema)
            stress_periods[name] = stress_period_data
        else:
            raise RuntimeError(
                "Package {}: invalid name, or package is not supported.".format(name)
            )
    # check if entire model is consumed
    assert (
        len(consumed_model) == 0
    ), "Model could not be completely written due to keys:{} ".format(
        list(consumed_model.keys())
    )

    runfile_parameters["packages"] = packages
    runfile_parameters["stress_periods"] = stress_periods
    # find number of species from length of "systems" within stress period packages
    runfile_parameters["ncomp"] = max(
        [len(pckg["conc"].keys()) for pckg in stress_periods.values()]
    )

    if times:
        runfile_parameters["time_discretisation"] = _time_discretisation(times)
    else:
        raise ValueError("No time dependent data in model.")

    return runfile_parameters


def seawat_write_runfile(path, runfile_parameters):
    """
    Writes an IMODSEAWAT runfile from metadata collected from model by
    `imod.run.get_runfile()`.

    Parameters
    ----------
    path : str
        Path to write runfile contents to.
    runfile_parameters : OrderedDict
        Dictionary used to fill in runfile.
    
    Returns
    -------
    None
    """
    template = _jinja2_template("seawat_runfile.j2")
    out = template.render(**runfile_parameters)

    with open(path, "w") as f:
        f.write(out)
