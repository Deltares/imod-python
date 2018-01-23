import numpy as np
from collections import OrderedDict
import xarray as xr
import re
import functools


def read_techeader(path):
    with open(path) as f:
        d = {}
        attrs = OrderedDict()
        coords = OrderedDict()

        line1 = f.readline()
        line1_parts = [part.strip().lower()
                       for part in line1.replace('"', '').split(',')]
        nvars = len(line1_parts) - 3
        d['data_vars'] = OrderedDict((var, None)
                                     for var in line1_parts[3:3 + nvars])

        line2 = f.readline()
        line2 = "".join(line2.split())  # remove all whitespace
        nlay = int(re.findall("K=\d+", line2)[0].split('=')[-1])
        nrow = int(re.findall("J=\d+", line2)[0].split('=')[-1])
        ncol = int(re.findall("I=\d+", line2)[0].split('=')[-1])
        coords['layer'] = np.arange(nlay)
        coords['row'] = np.arange(nrow)
        coords['column'] = np.arange(ncol)
        attrs['nlay'] = nlay
        attrs['nrow'] = nrow
        attrs['ncol'] = ncol
        d['coords'] = coords
        d['attrs'] = attrs
        return d


def get_time(line):
    line = "".join(line.split())  # remove all whitespace
    return np.float32(re.findall("ZONET=\"\d*.\d*", line)[0].split('="')[-1])


def determine_ntimes(nlines, count):
    ntimes = int(nlines / count)
    predicted_nlines = ntimes * (count + 2) + 1
    i = 0
    while predicted_nlines != nlines:
        if i > 100:
            raise RuntimeError("Could not find number of timesteps! "
                               "Check whether the TECPLOT file is well-formed.")
        ntimes += -1
        predicted_nlines = ntimes * (count + 2) + 1
        i += 1
    return ntimes


def index_lines(path):
    line_offset = []
    line_idx = []
    offset = 0
    idx = 0
    with open(path) as f:
        for line in f:
            line_offset.append(offset)
            line_idx.append(idx)
            offset += len(line)
            idx += len(line) + 1
    return line_offset, line_idx


def arr_to_dataset(arr, variables, time, **kwargs):
    nlay, nrow, ncol = [v for v in kwargs['attrs'].values()]
    kwargs['coords']['time'] = time
    for i, var in enumerate(list(kwargs['data_vars'].keys())):
        if var in variables:
            data = arr[:, i + 3].reshape(nlay, nrow, ncol)
            kwargs['data_vars'][var] = (('layer', 'row', 'column'), data)
        else:
            kwargs['data_vars'].pop(var)
    return xr.Dataset(**kwargs)


def load_tecfile(path, variables=None, times=None):
    """
    Loads the data from a TECPLOT file (.TEC), as outputted by iMODSEAWAT,
    into an xarray Dataset. The TECPLOT file provides no coordinate values,
    exclusively indices. The dataset is returned with dimensions: layer, row,
    column, time.

    Parameters
    ----------
    path: string
        path to .TEC file
    variables: list or tuple; optional
        Which variables to load into the xarray dataset, e.g:
        ['head', 'conc', 'vx', 'vy', 'vz']. Defaults to all variables.
    times: integer, list, or slice; optional
        Which timesteps to load. The TECPLOT file starts
        numbering at 0.0, and the numbers function solely as index.
        Defaults to all timesteps.

    Examples
    --------
    Load contents into an xarray dataset:
    >>> ds = load_tecfile(path)

    Load only head and conc data:
    >>> ds = load_tecfile(path, ['head','conc'])

    Load only vx data for the first and last timestep:
    >>> ds = load_tecfile(path, ['vx'], times=[0,-1])

    For the first 20 timesteps, once every four steps:
    >>> ds = load_tecfile(path, ['vx'], times=slice(0, 20, 4))

    Or for every tenth timestep:
    >>> ds = load_tecfile(path, ['vx'], times=slice(None, None, 10))

    See also the documentation for `slice()`.
    """
    tec_kwargs = read_techeader(path)
    line_offset, line_idx = index_lines(path)
    nlines = len(line_offset)
    nlines_timestep = functools.reduce(
        lambda x, y: x * y, [v for v in tec_kwargs['attrs'].values()])
    ntimes = determine_ntimes(nlines, nlines_timestep)

    if variables is None:
        variables = tec_kwargs['data_vars'].keys()
    else:
        variables = [var.lower() for var in variables]

    dss = []

    start_lines = [(t * (nlines_timestep + 2) + 1) for t in range(ntimes)]
    if times is None:
        pass
    else:
        lst = []
        lst.append(start_lines[times])  # always returns a list
        start_lines = lst  # also with only one element

    with open(path) as f:
        for start in start_lines:
            f.seek(line_idx[start])
            end = start + nlines_timestep
            time = get_time(f.readline())
            count = line_offset[end] - line_offset[start + 1]
            lines = f.readlines(count)
            arr = np.loadtxt(lines, np.float32, delimiter=',')
            dss.append(arr_to_dataset(arr, variables, time, **tec_kwargs))

    return xr.concat(dss, dim='time')
