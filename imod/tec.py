"""
Read Tecplot ASCII data.
"""

import collections
import functools
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr


def header(path):
    """Get the Tecplot ASCII header as a dictionary"""
    with open(path) as f:
        d = {}
        attrs = collections.OrderedDict()
        coords = collections.OrderedDict()

        line1 = f.readline()
        line1 = line1.split("=")[1]
        line1_parts = [
            part.strip().lower() for part in line1.replace('"', "").split(",")
        ]
        nvars = len(line1_parts) - 3
        d["coord_names"] = line1_parts[:3]
        d["data_vars"] = collections.OrderedDict(
            (var, None) for var in line1_parts[3 : 3 + nvars]
        )

        line2 = f.readline()
        line2 = "".join(line2.split())  # remove all whitespace
        nlay = int(re.findall(r"K=\d+", line2)[0].split("=")[-1])
        nrow = int(re.findall(r"J=\d+", line2)[0].split("=")[-1])
        ncol = int(re.findall(r"I=\d+", line2)[0].split("=")[-1])
        coords["z"] = np.arange(nlay)
        coords["y"] = np.arange(nrow)
        coords["x"] = np.arange(ncol)
        attrs["nlay"] = nlay
        attrs["nrow"] = nrow
        attrs["ncol"] = ncol
        d["coords"] = coords
        d["attrs"] = attrs
        return d


def _get_time(line):
    line = "".join(line.split())  # remove all whitespace
    return np.float32(re.match(r'ZONET="\D*(\d*[.]?\d*)', line)[1])


def _ntimes(nlines, count):
    ntimes = (nlines - 1.0) / (count + 2.0)
    if ntimes.is_integer():
        return int(ntimes)
    else:
        print(ntimes, nlines, count)
        raise RuntimeError(
            "Could not find number of timesteps!"
            "Check whether the Tecplot file is well-formed."
        )


def _vars_as_list(argument):
    """Type checking, returns a list for indexing."""
    if type(argument) in [list, tuple]:
        return argument
    elif type(argument) is str:
        return [argument]
    else:
        raise RuntimeError(
            "Invalid argument: accepts only lists, tuples, and" "strings."
        )


def _startlines_as_list(argument):
    """Type checking, returns a list for indexing."""
    if type(argument) is np.ndarray:
        return list(argument)
    elif type(argument) is np.int32:
        return [argument]


def _index_lines(path):
    line_idx = []
    idx = 0
    with open(path) as f:
        for line in f:
            line_idx.append(idx)
            idx += len(line) + 1
    return line_idx


def _dataset(df, time, **kwargs):
    nlay, nrow, ncol = [v for v in kwargs["attrs"].values()]
    d = {"x": ncol, "y": nrow, "z": nlay, "i": ncol, "j": nrow, "k": nlay}
    coords = tuple(kwargs.pop("coord_names")[::-1])  # flip order
    shape = tuple(d[c] for c in coords)
    kwargs["coords"]["time"] = time
    for c in coords:
        if c in df:
            kwargs["coords"][c] = df[c].unique()  # take sorting from tec
    for var in kwargs["data_vars"]:
        data = df[var].values.reshape(shape)
        kwargs["data_vars"][var] = (coords, data)
    return xr.Dataset(**kwargs)


def read(path, variables=None, times=None, kwargs={}):
    """
    Read a Tecplot ASCII data file to an xarray Dataset.

    Reads the data from a Tecplot ASCII file (.TEC or .DAT), as outputted by iMODSEAWAT,
    into an xarray Dataset. If there are valid coordinates x, y and z present in the Tecplot 
    file, these will be returned as coordinates (time, z, y, x). If the Tecplot file does not 
    provide coordinate values, only indices, then the dataset is returned with 
    dimensions: layer, row, column, time.

    Parameters
    ----------
    path: str or Path
        path to .TEC file
    variables: str, list, or tuple; optional
        Which variables to read into the xarray dataset, e.g:
        ['head', 'conc', 'vx', 'vy', 'vz']. Defaults to all variables.
    times: integer, list, or slice; optional
        Which timesteps to read. The Tecplot file starts
        numbering at 0.0, and the numbers function solely as index.
        Defaults to all timesteps.
    kwargs : dict
        Dictionary containing the ``pandas.read_csv()`` keyword arguments used
        for reading the Tecplot ASCII file (e.g. `{"delim_whitespace": True}`).

    Examples
    --------
    read contents into an xarray dataset:

    >>> ds = imod.tec.read(path)

    read only head and conc data:

    >>> ds = imod.tec.read(path, ['head','conc'])

    read only vx data for the first and last timestep:

    >>> ds = imod.tec.read(path, 'vx', times=[0,-1])

    For the first 20 timesteps, once every four steps:

    >>> ds = imod.tec.read(path, 'vx', times=slice(0, 20, 4))

    Or for every tenth timestep:
    
    >>> ds = imod.tec.read(path, 'vx', times=slice(None, None, 10))

    See also the documentation for ``slice()``.
    """
    # For a description of the Tecplot ASCII file format see:
    # ftp://ftp.tecplot.com/pub/doc/tecplot/360/dataformat.pdf
    tec_kwargs = header(path)

    # get a byte location for the start of every line
    # so that we can jump to locations in the file
    # and skip timesteps
    line_idx = _index_lines(path)
    nlines = len(line_idx)
    nlines_timestep = functools.reduce(
        lambda x, y: x * y, [v for v in tec_kwargs["attrs"].values()]
    )
    ntimes = _ntimes(nlines, nlines_timestep)

    if variables is None:
        variables = list(tec_kwargs["data_vars"].keys())
    else:
        variables = _vars_as_list(variables)
        variables = [var.lower() for var in variables]
    var_cols = [
        col_num
        for (col_num, var) in enumerate(tec_kwargs["data_vars"].keys())
        if var in variables
    ]
    variables = tec_kwargs["coord_names"] + variables
    var_cols = [0, 1, 2] + [3 + v for v in var_cols]
    for var in list(tec_kwargs["data_vars"].keys()):
        if var not in variables:
            tec_kwargs["data_vars"].pop(var)

    # generates a list of line numbers which indicate the start
    # of a new timestep
    start_lines = np.asarray([(t * (nlines_timestep + 2) + 1) for t in range(ntimes)])
    if times is None:
        pass
    else:
        start_lines = _startlines_as_list(start_lines[times])

    dss = []
    with open(path) as f:
        for start in start_lines:
            f.seek(line_idx[start])
            time = _get_time(f.readline())
            df = pd.read_csv(
                f, nrows=nlines_timestep, names=variables, usecols=var_cols, **kwargs
            )
            if start == start_lines[0]:
                # save coords columns for later, remove from further processing
                df_coords = df[tec_kwargs["coord_names"]]
                variables = variables[3:]
                var_cols = var_cols[3:]
            else:
                # append coords to df
                df = pd.concat((df_coords, df), axis=1)
            dss.append(_dataset(df, time, **tec_kwargs))
    dss = xr.concat(dss, dim="time")

    # check if coordinates mean something or are just indexes
    if (
        dss.coords["x"].values[0] == 1
        and dss.coords["x"].values[-1] == dss.attrs["ncol"]
    ):
        dss = dss.rename({"x": "column", "y": "row", "z": "layer"})
        return dss.transpose("time", "layer", "row", "column")
    else:
        # add layer coordinate
        dss = dss.assign_coords({"layer": ("z", range(1, dss.attrs["nlay"] + 1))})
        return dss.transpose("time", "z", "y", "x")
