import pathlib
import warnings

import numpy as np
import pandas as pd
import pkg_resources

# subpackages
import imod.evaluate
import imod.flow
import imod.mf6
import imod.prepare
import imod.select
import imod.visualize
import imod.wq

# submodules
from imod import idf, ipf, rasterio, run, tec, util
from imod.data_formats import gen

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass


def write(path, model, name=None, runfile_parameters=None):
    warnings.warn("imod.write is moved, use imod.flow.write instead.", FutureWarning)
    return imod.flow.write(
        path, model, name=name, runfile_parameters=runfile_parameters
    )


# remove together with imod.seawat_write
def _uniques(da):
    uniques = np.unique(da)
    return uniques[~np.isnan(uniques)]


# remove together with imod.seawat_write
def _all_single(a):
    return all([len(v) == 1 for v in a])


# remove together with imod.seawat_write
def _top_bot_dicts(model):
    top = model["top"]
    bot = model["bot"]
    layers = model["bnd"].coords["layer"].values
    unique_top = [_uniques(top.sel(layer=layer)) for layer in layers]
    unique_bot = [_uniques(bot.sel(layer=layer)) for layer in layers]
    # Check if there's a single valid value
    if _all_single(unique_top) and _all_single(unique_bot):
        d_tops = {layer: t[0] for layer, t in zip(layers, unique_top)}
        d_bots = {layer: b[0] for layer, b in zip(layers, unique_bot)}
        return d_tops, d_bots
    else:
        return None, None


def seawat_write(path, model, name=None, runfile_parameters=None):
    """
    Writes an iMODSEAWAT model, including runfile, as specified by ``model`` into
    directory ``path``.

    .. deprecated:: 0.7.0
        imod.seawat_write is deprecated, use the write method of imod.wq.SeawatModel instead.

    Directory ``path`` is created if it does not already exist.

    When ``runfile_parameters`` is specified, its values are used to fill in the
    runfile instead of those generated automatically from the
    data in ``model``. This is necessary when the default runfile parameters do
    not suffice, but you do not want to change the runfile after it is written.

    **Note**: every ``xarray.DataArray`` in ``model`` must have layer coordinates specified;
    use ``da.assign_coords(layer=...)``.

    Parameters
    ----------
    path : str
        The directory to write the model to.
    model : collections.OrderedDict
        Dictionary containing the package data as ``xarray.DataArray`` or
        ``pandas.DataFrame``.
    name : str
        Name given to the runfile. Defaults to "runfile".
    runfile_parameters : dict
        Dictionary containing the runfile parameters. Defaults to None,
        in which case runfile_parameters is generated from data in ``model``.

    Returns
    -------
    None

    Examples
    --------
    Write the model data in dictionary ``a`` as iMODFLOW model files, to directory
    "example_dir":

    >>> imod.write(path="example_dir", model=a)

    Generate runfile parameters for data in dictionary ``a`` using
    ``imod.run.get_runfile()``, change the value for ``hclose``, and write:

    >>> runfile_parameters = imod.run.get_runfile(model=a, seawat=True)
    >>> runfile_parameters["hclose"] = 0.00001
    >>> imod.seawat_write(path="example_dir", model=a, runfile_parameters=runfile_parameters)
    """
    warnings.warn(
        "imod.seawat_write is deprecated, use the write method of imod.wq.SeawatModel instead.",
        FutureWarning,
    )

    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)

    if runfile_parameters is None:
        runfile_parameters = imod.run.seawat_get_runfile(model, path)

    if name is None:
        name = "runfile"
    else:
        runfile_parameters["modelname"] = name

    runfile_path = path.joinpath("{}.run".format(name))
    imod.run.seawat_write_runfile(runfile_path, runfile_parameters)

    # Get data to write idf top and bot attributes  if possible:
    # when dz is constant over x, y.
    d_tops, d_bots = _top_bot_dicts(model)

    for key, data in model.items():
        # Get rid of time dimension
        # It might be present to force a transient run of otherwise steady-state
        # forcings.
        if key == "bnd":
            if "time" in data.dims:
                data = data.isel(time=0).drop("time")

        # Select the appropriate tops and bottoms for the present layers
        name = key.split("-")[0]
        package_path = path.joinpath(name).joinpath(key)
        if name == "wel" and isinstance(data, pd.DataFrame):
            if "time" in data.columns:
                imod.ipf.save(package_path, data, itype="timeseries")
            else:
                imod.ipf.save(package_path, data)
        else:
            if d_tops is not None:
                layers = np.atleast_1d(data.coords["layer"].values)
                data.attrs["top"] = [d_tops[layer] for layer in layers]
                data.attrs["bot"] = [d_bots[layer] for layer in layers]
            imod.idf.save(package_path, data)
