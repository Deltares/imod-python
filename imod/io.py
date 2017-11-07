import numpy as np
import numpy.ma as ma
from struct import unpack, pack
import pandas as pd
import xarray as xr
from glob import glob
from datetime import datetime
import os


# TODO, check this implementation with the format specification in the iMOD manual
def readipf(file):
    with open(file) as f:
        nrow = int(f.readline().strip())
        ncol = int(f.readline().strip())
        colnames = [f.readline().strip() for _ in range(ncol)]
        _ = f.readline()  # links to other files not handled
        df = pd.read_csv(f, header=None, names=colnames, nrows=nrow)
    return df


def writeipf(file, df):
    nrecords, nfields = df.shape
    with open(file, 'w') as f:
        f.write('{}\n{}\n'.format(nrecords, nfields))
        [f.write('{}\n'.format(colname)) for colname in list(df)]
        f.write('0,TXT\n')
        df.to_csv(f, index=False, header=False)


def readidf(file, nodata=None, header_only=False):
    # currently asserts ieq = ivf = 0, and comments are not read
    # meta is a dict that holds metadata stored in the IDF
    # metadata that can be derived from the data is not included
    meta = {}
    with open(file, 'rb') as f:
        assert unpack('i', f.read(4))[0] == 1271  # Lahey RecordLength Ident.
        ncol = unpack('i', f.read(4))[0]
        nrow = unpack('i', f.read(4))[0]
        meta['xmin'] = unpack('f', f.read(4))[0]
        meta['xmax'] = unpack('f', f.read(4))[0]
        meta['ymin'] = unpack('f', f.read(4))[0]
        meta['ymax'] = unpack('f', f.read(4))[0]
        f.read(4)  # minimum data value present
        f.read(4)  # maximum data value present
        nodataval = unpack('f', f.read(4))[0]
        meta['nodata'] = nodataval
        # only equidistant IDF currently supported
        assert not unpack('?', f.read(1))[0]  # ieq Bool
        itb = unpack('?', f.read(1))[0]
        meta['itb'] = itb
        # no usage of vectors currently supported
        assert not unpack('?', f.read(1))[0]  # ivf Bool
        f.read(1)  # not used
        meta['dx'] = unpack('f', f.read(4))[0]
        meta['dy'] = unpack('f', f.read(4))[0]
        if itb:
            meta['top'] = unpack('f', f.read(4))[0]
            meta['bot'] = unpack('f', f.read(4))[0]
        if header_only:
            meta['ncol'] = ncol
            meta['nrow'] = nrow
            return meta
        a = np.fromfile(f, np.float32, nrow * ncol).reshape((nrow, ncol))
    if nodata is None:
        return a, meta
    else:
        if np.isnan(nodataval):
            isnodata = np.isnan(a)
        else:
            isnodata = np.isclose(a, nodataval)
        if nodata == 'mask':
            return ma.masked_values(a, nodataval, copy=False), meta
        else:
            meta['nodata'] = nodata
            return np.where(isnodata, nodata, a), meta


def checkmeta(meta):
    required_keys = ('xmin', 'xmax', 'ymin', 'ymax',
                     'nodata', 'dx', 'dy')
    for key in required_keys:
        if key not in meta:
            raise ValueError('Meta dict needs to contain {}'.format(key))
    if meta.get('itb', False):  # assume itb is false if not in meta
        if ('top' not in meta) or ('bot' not in meta):
            raise ValueError(
                'Meta dict needs to contain top and bot if itb is True')


def writeidf(file, a, meta):
    # simplified writeidf function to writes bare numpy arrays
    # based on hardcoded metadata
    checkmeta(meta)  # do basic checks before opening the file
    with open(file, 'wb') as f:
        f.write(pack('i', 1271))  # Lahey RecordLength Ident.
        nrow, ncol = a.shape
        nodata = meta['nodata']
        itb = meta.get('itb', False)
        f.write(pack('i', ncol))
        f.write(pack('i', nrow))
        f.write(pack('f', meta['xmin']))
        f.write(pack('f', meta['xmax']))
        f.write(pack('f', meta['ymin']))
        f.write(pack('f', meta['ymax']))
        f.write(pack('f', a.min()))  # dmin
        f.write(pack('f', a.max()))  # dmax
        f.write(pack('f', nodata))
        f.write(pack('?', False))  # ieq
        f.write(pack('?', itb))
        f.write(pack('?', False))  # ivf
        f.write(pack('x'))  # not used
        f.write(pack('f', meta['dx']))
        f.write(pack('f', meta['dy']))
        if itb:
            f.write(pack('f', meta['top']))
            f.write(pack('f', meta['bot']))
        # convert to a ndarray of float32
        # TODO convert np.nan in ndarray to nodata as well
        if isinstance(a, ma.MaskedArray):
            a = a.filled(nodata)
        elif isinstance(a, xr.DataArray):
            a = a.fillna(nodata).values
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        a.tofile(f)


def parse_filename(path):
    """Parse a filename, returning a dict of the parts,
    following the iMOD conventions"""
    noext = os.path.splitext(path)[0]
    parts = os.path.basename(noext).split('_')
    param = parts[0]
    d = {'param': param}
    try:
        d['date'] = np.datetime64(datetime.strptime(parts[1], '%Y%m%d%H%M%S'))
    except ValueError:
        pass  # no date in dict
    # layer is always last
    if parts[-1].lower().startswith(b'l'):
        d['layer'] = int(parts[-1][1:])
    return d


def loaddata(result_dir, filename_pattern, time=True):
    globpath = os.path.join(result_dir, filename_pattern)
    paths = glob(globpath)

    # prepare DataArray coordinates
    meta = readidf(paths[0], header_only=True)
    if time:
        dates = np.unique([parse_filename(path)['date'] for path in paths])
        nper = dates.size
    else:
        dates = None
        nper = None
    layers = np.unique([parse_filename(path)['layer'] for path in paths])
    xcoords = np.arange(meta['xmin'] + meta['dx'] / 2.0,
                        meta['xmax'],
                        meta['dx'])
    ycoords = np.arange(meta['ymin'] + meta['dy'] / 2.0,
                        meta['ymax'],
                        meta['dy'])
    nlay = layers.size
    nrow = meta['nrow']
    ncol = meta['ncol']

    coords = {'layer': layers,
              'row': range(1, nrow + 1),
              'column': range(1, ncol + 1),
              # non-dimension coordinates
              'y': ('row', ycoords),
              'x': ('column', xcoords)}
    # allocate arrays to store idf grids
    if time:
        coords['time'] = dates
        dims = ('time', 'layer', 'row', 'column')
        values = np.zeros((nper, nlay, nrow, ncol), dtype=np.float32)
    else:
        dims = ('layer', 'row', 'column')
        values = np.zeros((nlay, nrow, ncol), dtype=np.float32)
    data = xr.DataArray(values,
                        coords=coords,
                        dims=dims)
    # make row and column immutable indexes, such that we can use ds.sel(row=2)
    # https://github.com/pydata/xarray/issues/934#issuecomment-236960237
    # not working, yet, for now we just make row and column indexes and x and y coordinates
    # data['row'].to_index()
    # data['column'].to_index()
    # load in idf data
    for path in paths:
        fndict = parse_filename(path)
        arr, _ = readidf(path, nodata=np.nan)
        layer = fndict['layer']
        layeridx = layers.searchsorted(layer)
        # not all are always present
        if time:
            date = fndict['date']
            dateidx = dates.searchsorted(date)
            data[dateidx, layeridx, :, :] = arr
        else:
            data[layeridx, :, :] = arr
    return data


# xarray: if your data is unstructured or one-dimensional, stick with pandas
# pandas.Panel is deprecated, use MultiIndex DataFrame or xarray
def loadipf(ipfdir, filename_pattern):
    globpath = os.path.join(ipfdir, filename_pattern)
    paths = glob(globpath)

    dfs = []
    for path in paths:
        layer = parse_filename(path)['layer']
        df = readipf(path)
        df['layer'] = layer
        dfs.append(df)

    bigdf = pd.concat(dfs, ignore_index=True)
    return bigdf
