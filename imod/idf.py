import os
import numpy as np
import rasterio
from struct import unpack, pack
from collections import OrderedDict
import pandas as pd
import xarray as xr
from dask import array
from glob import glob
from datetime import datetime
from imod import util


def header(path):
    attrs = util.decompose(path)
    with open(path, 'rb') as f:
        assert unpack('i', f.read(4))[0] == 1271  # Lahey RecordLength Ident.
        ncol = unpack('i', f.read(4))[0]
        nrow = unpack('i', f.read(4))[0]
        xmin = unpack('f', f.read(4))[0]
        f.read(4)  # xmax
        f.read(4)  # ymin
        ymax = unpack('f', f.read(4))[0]
        # note that dmin and dmax are currently not kept up to date
        # generally ok, they are only used for legends in iMOD
        # but would be nice if we could find a nice way to do this
        f.read(4)  # dmin, minimum data value present
        f.read(4)  # dmax, maximum data value present
        nodata = unpack('f', f.read(4))[0]
        attrs['nodata'] = nodata
        # only equidistant IDF currently supported
        assert not unpack('?', f.read(1))[0]  # ieq Bool
        itb = unpack('?', f.read(1))[0]
        # no usage of vectors currently supported
        assert not unpack('?', f.read(1))[0]  # ivf Bool
        f.read(1)  # not used
        cellwidth = unpack('f', f.read(4))[0]
        cellheight = unpack('f', f.read(4))[0]
        # res is always positive, this seems to be the rasterio behavior
        attrs['res'] = (cellwidth, cellheight)
        # xarray converts affine to tuple, so we follow that
        # TODO change after https://github.com/pydata/xarray/pull/1712 is released
        attrs['transform'] = (cellwidth, 0.0, xmin, 0.0, -cellheight, ymax)
        if itb:
            attrs['top'] = unpack('f', f.read(4))[0]
            attrs['bot'] = unpack('f', f.read(4))[0]

        # These are derived, remove after using them downstream
        attrs['headersize'] = f.tell()
        attrs['ncol'] = ncol
        attrs['nrow'] = nrow
    return attrs


def setnodataheader(path, nodata):
    with open(path, 'r+b') as f:
        f.seek(36)  # go to nodata position
        f.write(pack('f', nodata))


def _pre_data_read(path):
    # currently asserts ieq = ivf = 0, and comments are not read
    attrs = header(path)
    headersize = attrs.pop('headersize')
    return attrs, headersize


def _to_nan(a, attrs):
    # always convert nodata values to NaN, this is how xarray deals with it
    nodata = attrs.pop('nodata')
    if np.isnan(nodata):
        return a, attrs
    else:
        isnodata = np.isclose(a, nodata)
        a[isnodata] = np.nan
        return a, attrs


def memmap(path):
    attrs, headersize = _pre_data_read(path)
    a = np.memmap(path, np.float32, 'r+', headersize,
                  (attrs['nrow'], attrs['ncol']))
    setnodataheader(path, np.nan)
    return _to_nan(a, attrs)


def read(path):
    attrs, headersize = _pre_data_read(path)
    with open(path, 'rb') as f:
        f.seek(headersize)
        a = np.reshape(np.fromfile(
            f, np.float32, attrs['nrow'] * attrs['ncol']), (attrs['nrow'], attrs['ncol']))
    return _to_nan(a, attrs)


def dask(path, chunks=None, memmap=True):
    if memmap:
        a, attrs = memmap(path)
    else:
        a, attrs = read(path)
    # grab the whole array as one chunk
    if chunks is None:
        chunks = a.shape
    x = array.from_array(a, chunks=chunks)
    return x, attrs


def _dataarray_kwargs(path, attrs):
    """Construct xarray coordinates from
    IDF filename and attrs dict"""
    attrs.update(util.decompose(path))
    name = attrs.pop('name')  # avoid storing information twice
    d = {
        'name': name,
        'dims': ('y', 'x'),  # only two dimensions in a single IDF
        'attrs': attrs,
    }

    # add the available coordinates
    coords = OrderedDict()

    # dimension coordinates
    nrow = attrs.pop('nrow')
    ncol = attrs.pop('ncol')
    dx = attrs['transform'][0]  # always positive
    xmin = attrs['transform'][2]
    dy = attrs['transform'][4]  # always negative
    ymax = attrs['transform'][5]
    xmax = xmin + ncol * dx
    ymin = ymax + nrow * dy
    xcoords = np.arange(xmin + dx / 2.0, xmax, dx)
    ycoords = np.arange(ymax + dy / 2.0, ymin, dy)
    coords['y'] = ycoords
    coords['x'] = xcoords

    # these will become dimension coordinates when combining IDFs
    layer = attrs.pop('layer', None)
    if layer is not None:
        coords['layer'] = layer

    time = attrs.pop('time', None)
    if time is not None:
        coords['time'] = time

    d['coords'] = coords

    return d


def dataarray(path, chunks=None, memmap=True):
    x, attrs = dask(path, chunks=chunks, memmap=memmap)
    kwargs = _dataarray_kwargs(path, attrs)
    return xr.DataArray(x, **kwargs)


# load IDFs for multiple times and/or layers into one DataArray
def load(path, chunks=None, memmap=True):
    if isinstance(path, list):
        return _load_list(path, chunks=chunks, memmap=memmap)
    paths = glob(path)
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(
            'Could not find any files matching {}'.format(path))
    elif n == 1:
        return dataarray(paths[0], chunks=chunks, memmap=memmap)
    return _load_list(paths, chunks=chunks, memmap=memmap)


def _load_list(paths, chunks=None, memmap=True):
    # create a DataArray from every IDF
    das = [dataarray(path, chunks=chunks, memmap=memmap) for path in paths]
    assert all(
        da.name == das[0].name for da in das), "DataArrays to be combined need to have the same name"
    # combine the different DataArrays into one DataArray with added dimensions

    # xarray currently does not seem to be able to automatically combine the
    # different coords into new dimensions, so we do it manually instead
    # unfortunately that means we only support adding the 'layer' and 'time' dimensions
    da0 = das[0]  # this should apply to all the same
    haslayer = 'layer' in da0.coords
    hastime = 'time' in da0.coords
    if haslayer:
        nlayer = np.unique([da.layer.values for da in das]).size
        if hastime:
            ntime = np.unique([da.time.values for da in das]).size
            das.sort(key=lambda da: (da.time, da.layer))
            # first create the layer dimension for each time
            das_layer = []
            s, e = 0, nlayer
            for i in range(ntime):
                das_layer.append(xr.concat(das[s:e], dim='layer'))
                s = e
                e = s + nlayer
            # then add the time dimension on top of that
            da = xr.concat(das_layer, dim='time')
        else:
            das.sort(key=lambda da: da.layer)
            da = xr.concat(das, dim='layer')
    else:
        if hastime:
            das.sort(key=lambda da: da.time)
            da = xr.concat(das, dim='time')
        else:
            assert(len(das) == 1)
            da = das[0]

    return da


def loadset(globpath, chunks=None, memmap=True):
    # recursively find all files, use ** in globpath to indicate where
    # e.g. globpath = 'model/**/*.idf'
    paths = glob(globpath, recursive=True)
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(
            'Could not find any files matching {}'.format(globpath))
    # group the DataArrays together using their name
    # note that directory names are ignored, and in case of duplicates, the last one wins
    names = [util.decompose(path)['name'] for path in paths]
    unique_names = list(np.unique(names))
    d = OrderedDict()
    for n in unique_names:
        d[n] = []  # prepare empty lists to append to
    for p, n in zip(paths, names):
        d[n].append(p)

    # load each group into a DataArray
    das = [_load_list(v, chunks=chunks, memmap=memmap) for v in d.values()]

    # store each DataArray under it's own name in an OrderedDict
    dd = OrderedDict()
    for da in das:
        dd[da.name] = da
    # Initially I wanted to return a xarray Dataset here,
    # but then realised that it is not always aligned, and therefore not possible, see
    # https://github.com/pydata/xarray/issues/1471#issuecomment-313719395
    # It is not aligned when some parameters only have a non empty subset of a dimension,
    # such as L2 + L3. This dict provides a similar interface anyway. If a Dataset is constructed
    # from unaligned DataArrays it will make copies of the memmap, which we don't want.
    return dd


# write DataArrays to IDF
def save(dirpath, a):
    d = {'extension': 'idf'}
    if a.name is None:
        raise ValueError("DataArray name cannot be None")
    else:
        d['name'] = a.name
    os.makedirs(dirpath, exist_ok=True)
    if 'time' in a.coords:
        # TODO implement (not much different than layer)
        raise NotImplementedError(
            "Writing time dependent IDFs not yet implemented")
    if 'layer' in a.coords:
        if 'layer' in a.dims:
            for layer, a2d in a.groupby('layer'):
                d['layer'] = layer
                path = os.path.join(dirpath, util.compose(d))
                write(path, a2d)
        else:
            d['layer'] = int(a.coords['layer'])
            path = os.path.join(dirpath, util.compose(d))
            write(path, a)
    else:
        path = os.path.join(dirpath, util.compose(d))
        write(path, a)


def write(path, a):
    assert(a.dims == ('y', 'x'))
    with open(path, 'wb') as f:
        f.write(pack('i', 1271))  # Lahey RecordLength Ident.
        nrow = a.y.size
        ncol = a.x.size
        nodata = np.nan
        attrs = a.attrs
        itb = attrs.get('top', False) and attrs.get('bot', False)
        f.write(pack('i', ncol))
        f.write(pack('i', nrow))
        # the attribute is simply a 9 tuple
        transform = rasterio.Affine(*attrs['transform'][:6])
        xmin, ymin, xmax, ymax = rasterio.transform.array_bounds(
            nrow, ncol, transform)
        f.write(pack('f', xmin))
        f.write(pack('f', xmax))
        f.write(pack('f', ymin))
        f.write(pack('f', ymax))
        f.write(pack('f', float(a.min())))  # dmin
        f.write(pack('f', float(a.max())))  # dmax
        f.write(pack('f', nodata))
        f.write(pack('?', False))  # ieq
        f.write(pack('?', itb))
        f.write(pack('?', False))  # ivf
        f.write(pack('x'))  # not used
        f.write(pack('f', attrs['res'][0]))
        f.write(pack('f', attrs['res'][1]))
        if itb:
            f.write(pack('f', attrs['top']))
            f.write(pack('f', attrs['bot']))
        # convert to a ndarray of float32
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        a.values.tofile(f)
