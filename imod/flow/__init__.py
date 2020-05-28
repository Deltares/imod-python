"""
Create iMODFLOW model.

With this module full iMODFLOW models can be written to disk. In :mod:`imod.wq`
this is done with classes that understand more about the model and can therefore
offer more help. Here a more rudimentary approach is used where you can pass a
dictionary with all data. This works, but you need to be extra careful about
writing a full and correct model definition, since no extra checks are done by
this module.
"""

import pathlib
import warnings

import numpy as np
import pandas as pd

import imod


def write(path, model, name=None, runfile_parameters=None, output_packs=["shd"]):
    """
    Writes an iMODFLOW model, including runfile, as specified by ``model`` into
    directory ``path``.

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

    >>> imod.flow.write(path="example_dir", model=a)

    Generate runfile parameters for data in dictionary ``a`` using
    ``imod.run.get_runfile()``, change the value for ``hclose``, and write:

    >>> runfile_parameters = imod.run.get_runfile(model=a)
    >>> runfile_parameters["hclose"] = 0.00001
    >>> imod.flow.write(path="example_dir", model=a, runfile_parameters=runfile_parameters)
    """
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)

    if runfile_parameters is None:
        runfile_parameters = imod.run.get_runfile(model, path, output_packs)

    if name is None:
        name = "runfile"
    else:
        runfile_parameters["modelname"] = name

    runfile_path = path.joinpath("{}.run".format(name))
    imod.run.write_runfile(runfile_path, runfile_parameters)

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
                imod.ipf.save(package_path, data, itype="timeseries")
            else:
                imod.ipf.save(package_path, data)
        else:
            imod.idf.save(package_path, data)
