from pathlib import Path

# without these this usage wouldn't work:
# import imod
# imod.idf.*
from imod import idf
from imod import ipf
from imod import tec
from imod import util
from imod import run

# since this is a big dependency that is sometimes hard to install
# and not always required, we made this an optional dependency
try:
    from imod import rasterio
except ImportError:
    pass

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


def write(path, model, name=None, runfile_parameters=None):
    """
    Writes an iMODFLOW model, including runfile, as specified by `model` into 
    directory `path`.

    Directory `path` is created if it does not already exist.

    When `runfile_parameters` is specified, its values are used to fill in the
    runfile instead of those generated automatically from the 
    data in `model`. This is necessary when the default runfile parameters do 
    not suffice, but you do not want to change the runfile after it is written.

    **Note**: every `xarray.DataArray` in `model` must have layer coordinates specified;
    use `da.assign_coords(layer=...)`.

    Parameters
    ----------
    path : str
        The directory to write the model to.
    model : OrderedDict
        `OrderedDict` containing the package data as `xarray.DataArray` or
        `pandas.DataFrame`.
    name : str
        Name given to the runfile. Defaults to "runfile".
    runfile_parameters : dict
        Dictionary containing the runfile parameters. Defaults to None,
        in which case runfile_parameters is generated from data in `model`.
    
    Returns
    -------
    None

    Examples
    --------
    Write the model data in dictionary `a` as iMODFLOW model files, to directory 
    "example_dir":

    >>> imod.write(path="example_dir", model=a)

    Generate runfile parameters for data in dictionary `a` using
    `imod.run.get_runfile()`, change the value for `hclose`, and write:

    >>> runfile_parameters = imod.run.get_runfile(model=a)
    >>> runfile_parameters["hclose"] = 0.00001
    >>> imod.write(path="example_dir", model=a, runfile_parameters=runfile_parameters)

    """ 
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    if runfile_parameters is None:
        runfile_parameters = run.get_runfile(model, path)

    if name is None:
        name = "runfile"
    else:
        runfile_parameters["modelname"] = name

    runfile_path = path.joinpath("{}.run".format(name))
    run.write_runfile(runfile_path, runfile_parameters)

    for key, data in model.items():
        # Get rid of time dimension
        # It might be present to force a transient run of otherwise steady-state
        # forcings.
        if key == "bnd":
            if "time" in data.dims:
                data = data.isel(time=0).drop("time")

        name = key.split("-")[0]
        package_path = path.joinpath(name).joinpath(key)
        if name == "wel":
            if "time" in data.columns:
                ipf.save(package_path, data, itype="timeseries")
            else:
                ipf.save(package_path, data)
        else:
            idf.save(package_path, data)


def seawat_write(path, model, name=None, runfile_parameters=None):
    """
    Writes an iMODSEAWAT model, including runfile, as specified by `model` into 
    directory `path`.

    Directory `path` is created if it does not already exist.

    When `runfile_parameters` is specified, its values are used to fill in the
    runfile instead of those generated automatically from the 
    data in `model`. This is necessary when the default runfile parameters do 
    not suffice, but you do not want to change the runfile after it is written.

    **Note**: every `xarray.DataArray` in `model` must have layer coordinates specified;
    use `da.assign_coords(layer=...)`.

    Parameters
    ----------
    path : str
        The directory to write the model to.
    model : OrderedDict
        `OrderedDict` containing the package data as `xarray.DataArray` or
        `pandas.DataFrame`.
    name : str
        Name given to the runfile. Defaults to "runfile".
    runfile_parameters : dict
        Dictionary containing the runfile parameters. Defaults to None,
        in which case runfile_parameters is generated from data in `model`.
    
    Returns
    -------
    None

    Examples
    --------
    Write the model data in dictionary `a` as iMODFLOW model files, to directory 
    "example_dir":

    >>> imod.write(path="example_dir", model=a)

    Generate runfile parameters for data in dictionary `a` using
    `imod.run.get_runfile()`, change the value for `hclose`, and write:

    >>> runfile_parameters = imod.run.get_runfile(model=a, seawat=True)
    >>> runfile_parameters["hclose"] = 0.00001
    >>> imod.seawat_write(path="example_dir", model=a, runfile_parameters=runfile_parameters)

    """ 
    # TODO: does this really belong here?
    import pandas as pd
    
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    if runfile_parameters is None:
        runfile_parameters = run.seawat_get_runfile(model, path)

    if name is None:
        name = "runfile"
    else:
        runfile_parameters["modelname"] = name

    runfile_path = path.joinpath("{}.run".format(name))
    run.seawat_write_runfile(runfile_path, runfile_parameters)

    for key, data in model.items():
        # Get rid of time dimension
        # It might be present to force a transient run of otherwise steady-state
        # forcings.
        if key == "bnd":
            if "time" in data.dims:
                data = data.isel(time=0).drop("time")
        
        name = key.split("-")[0]
        package_path = path.joinpath(name).joinpath(key)
        if name == "wel" and isinstance(data, pd.DataFrame):
            if "time" in data.columns:
                ipf.save(package_path, data, itype="timeseries")
            else:
                ipf.save(package_path, data)
        else:
            idf.save(package_path, data)