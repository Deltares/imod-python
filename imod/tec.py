import numpy as np
from collections import OrderedDict
import xarray as xr
import re
import functools
import pandas as pd


def header(path):
    """Get the Tecplot ASCII header as a dictionary"""
    with open(path) as f:
        d = {}
        attrs = OrderedDict()
        coords = OrderedDict()

        line1 = f.readline()
        line1_parts = [
            part.strip().lower() for part in line1.replace('"', "").split(",")
        ]
        nvars = len(line1_parts) - 3
        d["data_vars"] = OrderedDict((var, None) for var in line1_parts[3 : 3 + nvars])

        line2 = f.readline()
        line2 = "".join(line2.split())  # remove all whitespace
        nlay = int(re.findall(r"K=\d+", line2)[0].split("=")[-1])
        nrow = int(re.findall(r"J=\d+", line2)[0].split("=")[-1])
        ncol = int(re.findall(r"I=\d+", line2)[0].split("=")[-1])
        coords["layer"] = np.arange(nlay)
        coords["row"] = np.arange(nrow)
        coords["column"] = np.arange(ncol)
        attrs["nlay"] = nlay
        attrs["nrow"] = nrow
        attrs["ncol"] = ncol
        attrs["nvars"] = nvars
        d["coords"] = coords
        d["attrs"] = attrs
        return d


def _get_time(line):
    line = "".join(line.split())  # remove all whitespace
    return np.float32(re.findall(r'ZONET="\d*.\d*', line)[0].split('="')[-1])


def _ntimes(nlines, count):
    ntimes = (nlines - 1.0) / (count + 2.0)
    if ntimes.is_integer():
        return int(ntimes)
    else:
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
    kwargs["coords"]["time"] = time
    for var in df:
        data = df[var].values.reshape(nlay, nrow, ncol)
        kwargs["data_vars"][var] = (("layer", "row", "column"), data)
    return xr.Dataset(**kwargs)


def load(path, variables=None, times=None):
    """Load a Tecplot ASCII data file to an xarray Dataset.

    Loads the data from a Tecplot ASCII file (.TEC or .DAT), as outputted by iMODSEAWAT,
    into an xarray Dataset. The Tecplot file provides no coordinate values,
    exclusively indices. The dataset is returned with dimensions: layer, row,
    column, time.

    Parameters
    ----------
    path: str or Path
        path to .TEC file
    variables: str, list, or tuple; optional
        Which variables to load into the xarray dataset, e.g:
        ['head', 'conc', 'vx', 'vy', 'vz']. Defaults to all variables.
    times: integer, list, or slice; optional
        Which timesteps to load. The Tecplot file starts
        numbering at 0.0, and the numbers function solely as index.
        Defaults to all timesteps.

    Examples
    --------
    Load contents into an xarray dataset:

    >>> ds = load(path)

    Load only head and conc data:

    >>> ds = load(path, ['head','conc'])

    Load only vx data for the first and last timestep:

    >>> ds = load(path, 'vx', times=[0,-1])

    For the first 20 timesteps, once every four steps:

    >>> ds = load(path, 'vx', times=slice(0, 20, 4))

    Or for every tenth timestep:
    
    >>> ds = load(path, 'vx', times=slice(None, None, 10))

    See also the documentation for `slice()`.
    """
    # For a description of the Tecplot ASCII file format see:
    # ftp://ftp.tecplot.com/pub/doc/tecplot/360/dataformat.pdf
    tec_kwargs = header(path)
    var_cols = range(3, 3 + tec_kwargs["attrs"].pop("nvars"))
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
            for (col_num, var) in zip(var_cols, tec_kwargs["data_vars"].keys())
            if var in variables
        ]
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
                f, nrows=nlines_timestep, names=variables, usecols=var_cols
            )
            dss.append(_dataset(df, time, **tec_kwargs))

    return xr.concat(dss, dim="time")
