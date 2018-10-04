import numpy as np
from struct import unpack, pack
from collections import OrderedDict
import pandas as pd
import xarray as xr
from dask import array
from glob import glob
from pathlib import Path
from datetime import datetime
import imod
from imod import util


def header(path):
    """Read the IDF header information into a dictionary"""
    attrs = util.decompose(path)
    with open(path, "rb") as f:
        assert unpack("i", f.read(4))[0] == 1271  # Lahey RecordLength Ident.
        ncol = unpack("i", f.read(4))[0]
        nrow = unpack("i", f.read(4))[0]
        xmin = unpack("f", f.read(4))[0]
        f.read(4)  # xmax
        f.read(4)  # ymin
        ymax = unpack("f", f.read(4))[0]
        # note that dmin and dmax are currently not kept up to date
        # generally ok, they are only used for legends in iMOD
        # but would be nice if we could find a nice way to do this
        f.read(4)  # dmin, minimum data value present
        f.read(4)  # dmax, maximum data value present
        nodata = unpack("f", f.read(4))[0]
        attrs["nodata"] = nodata
        # only equidistant IDF currently supported
        assert not unpack("?", f.read(1))[0]  # ieq Bool
        itb = unpack("?", f.read(1))[0]
        # no usage of vectors currently supported
        assert not unpack("?", f.read(1))[0]  # ivf Bool
        f.read(1)  # not used
        cellwidth = unpack("f", f.read(4))[0]
        cellheight = unpack("f", f.read(4))[0]
        # res is always positive, this seems to be the rasterio behavior
        attrs["res"] = (cellwidth, cellheight)
        # xarray converts affine to tuple, so we follow that
        # TODO change after https://github.com/pydata/xarray/pull/1712 is released
        attrs["transform"] = (cellwidth, 0.0, xmin, 0.0, -cellheight, ymax)
        if itb:
            attrs["top"] = unpack("f", f.read(4))[0]
            attrs["bot"] = unpack("f", f.read(4))[0]

        # These are derived, remove after using them downstream
        attrs["headersize"] = f.tell()
        attrs["ncol"] = ncol
        attrs["nrow"] = nrow
    return attrs


def setnodataheader(path, nodata):
    """Change the nodata value in the IDF header"""
    with open(path, "r+b") as f:
        f.seek(36)  # go to nodata position
        f.write(pack("f", nodata))


def _pre_data_read(path):
    # shared code for idf.read and idf.memmap
    # currently asserts ieq = ivf = 0, and comments are not read
    attrs = header(path)
    headersize = attrs.pop("headersize")
    return attrs, headersize


def _to_nan(a, attrs):
    """Change all nodata values in the array to NaN"""
    # it needs to be NaN for xarray to deal with it properly
    # no need to store the nodata value if it is always NaN
    nodata = attrs.pop("nodata")
    if np.isnan(nodata):
        return a, attrs
    else:
        isnodata = np.isclose(a, nodata)
        a[isnodata] = np.nan
        return a, attrs


def memmap(path):
    """Make a memory map of a single IDF file
    
    Use idf.read if you don't want to modify the nodata values of the IDF,
    or if you want to have an in memory numpy.ndarray.

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be memory mapped
    
    Returns
    -------
    numpy.memmap
        A float32 memory map with shape (nrow, ncol) of the data block
        of the IDF file. It is opened in 'r+' read and write mode, and
        on opening all nodata values are changed to NaN in both the
        header and data block.
    dict
        A dict with all metadata.
    """
    attrs, headersize = _pre_data_read(path)
    a = np.memmap(
        str(path), np.float32, "r+", headersize, (attrs["nrow"], attrs["ncol"])
    )

    # only change the header if needed
    if not np.isnan(attrs["nodata"]):
        setnodataheader(path, np.nan)

    return _to_nan(a, attrs)


def read(path):
    """Read a single IDF file to a numpy.ndarray
    
    Compared to idf.memmap, this does not modify the IDF.

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be read
    
    Returns
    -------
    numpy.ndarray
        A float32 numpy.ndarray with shape (nrow, ncol) of the values
        in the IDF file. On opening all nodata values are changed
        to NaN in the numpy.ndarray.
    dict
        A dict with all metadata.
    """
    attrs, headersize = _pre_data_read(path)
    with open(path, "rb") as f:
        f.seek(headersize)
        a = np.reshape(
            np.fromfile(f, np.float32, attrs["nrow"] * attrs["ncol"]),
            (attrs["nrow"], attrs["ncol"]),
        )
    return _to_nan(a, attrs)


def dask(path, chunks=None, memmap=False):
    """Read a single IDF file to a dask.array
    
    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be read
    chunks : int or tuple of int, optional
        How to chunk the array. By default it creates only 1 chunk.
    memmap : bool, optional
        Whether to use a memory map to the file, or an in memory
        copy. Default is to use a memory map.
    
    Returns
    -------
    dask.array
        A float32 dask.array with shape (nrow, ncol) of the values
        in the IDF file. On opening all nodata values are changed
        to NaN in the dask.array.
    dict
        A dict with all metadata.
    """
    if memmap:
        a, attrs = imod.idf.memmap(path)
    else:
        a, attrs = imod.idf.read(path)
    # grab the whole array as one chunk
    if chunks is None:
        chunks = a.shape
    x = array.from_array(a, chunks=chunks)
    return x, attrs


def _dataarray_kwargs(path, attrs):
    """Construct xarray coordinates from IDF filename and attrs dict"""
    attrs.update(util.decompose(path))
    # from decompose, but not needed in attrs
    attrs.pop("directory")
    attrs.pop("extension")
    name = attrs.pop("name")  # avoid storing information twice
    d = {
        "name": name,
        "dims": ("y", "x"),  # only two dimensions in a single IDF
        "attrs": attrs,
    }

    # add the available coordinates
    coords = OrderedDict()

    # dimension coordinates
    nrow = attrs.pop("nrow")
    ncol = attrs.pop("ncol")
    dx = attrs["transform"][0]  # always positive
    xmin = attrs["transform"][2]
    dy = attrs["transform"][4]  # always negative
    ymax = attrs["transform"][5]
    xmax = xmin + ncol * dx
    ymin = ymax + nrow * dy
    xcoords = np.arange(xmin + dx / 2.0, xmax, dx)
    ycoords = np.arange(ymax + dy / 2.0, ymin, dy)
    coords["y"] = ycoords
    coords["x"] = xcoords

    # these will become dimension coordinates when combining IDFs
    layer = attrs.pop("layer", None)
    if layer is not None:
        coords["layer"] = layer

    time = attrs.pop("time", None)
    if time is not None:
        coords["time"] = time

    d["coords"] = coords

    return d


def dataarray(path, chunks=None, memmap=False):
    """Read a single IDF file to a xarray.DataArray
    
    The function imod.idf.load is more general and can load multiple layers
    and/or timestamps at once.

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be read
    chunks : int or tuple of int, optional
        How to chunk the array. By default it creates only 1 chunk.
    memmap : bool, optional
        Whether to use a memory map to the file, or an in memory
        copy. Default is to use a memory map.
    
    Returns
    -------
    xarray.DataArray
        A float32 xarray.DataArray of the values in the IDF file.
        All metadata needed for writing the file to IDF or other formats
        using imod.rasterio are included in the xarray.DataArray.attrs.
    """
    x, attrs = dask(path, chunks=chunks, memmap=memmap)
    kwargs = _dataarray_kwargs(path, attrs)
    return xr.DataArray(x, **kwargs)


# load IDFs for multiple times and/or layers into one DataArray
def load(path, chunks=None, memmap=False):
    """Read a parameter (one or more IDFs) to a xarray.DataArray
    
    Parameters
    ----------
    path : str, Path or list
        This can be a single file, 'head_l1.idf', a glob pattern expansion,
        'head_l*.idf', or a list of files, ['head_l1.idf', 'head_l2.idf'].
        Note that each file needs to be of the same name (part before the 
        first underscore) but have a different layer and/or timestamp,
        such that they can be combined in a single xarray.DataArray.
    chunks : int or tuple of int, optional
        How to chunk the array. By default it creates only 1 chunk.
    memmap : bool, optional
        Whether to use a memory map to the file, or an in memory
        copy. Default is to use a memory map.
    
    Returns
    -------
    xarray.DataArray
        A float32 xarray.DataArray of the values in the IDF file(s).
        All metadata needed for writing the file to IDF or other formats
        using imod.rasterio are included in the xarray.DataArray.attrs.
    """
    if isinstance(path, list):
        return _load_list(path, chunks=chunks, memmap=memmap)
    elif isinstance(path, Path):
        path = str(path)

    paths = [Path(p) for p in glob(path)]
    n = len(paths)
    if n == 0:
        raise FileNotFoundError("Could not find any files matching {}".format(path))
    elif n == 1:
        return dataarray(paths[0], chunks=chunks, memmap=memmap)
    return _load_list(paths, chunks=chunks, memmap=memmap)


def _load_list(paths, chunks=None, memmap=False):
    """Combine a list of paths to IDFs to a single xarray.DataArray"""
    # first load every IDF into a separate DataArray
    das = [dataarray(path, chunks=chunks, memmap=memmap) for path in paths]
    assert all(
        da.name == das[0].name for da in das
    ), "DataArrays to be combined need to have the same name"

    # combine the different DataArrays into one DataArray with added dimensions
    # xarray currently does not seem to be able to automatically combine the
    # different coords into new dimensions, so we do it manually instead
    # unfortunately that means we only support adding the 'layer' and 'time' dimensions
    da0 = das[0]  # this should apply to all the same
    haslayer = "layer" in da0.coords
    hastime = "time" in da0.coords
    if haslayer:
        nlayer = np.unique([da.coords["layer"].values for da in das]).size
        if hastime:
            ntime = np.unique([da.coords["time"].values for da in das]).size
            das.sort(key=lambda da: (da.coords["time"], da.coords["layer"]))
            # first create the layer dimension for each time
            das_layer = []
            s, e = 0, nlayer
            for _ in range(ntime):
                das_layer.append(xr.concat(das[s:e], dim="layer"))
                s = e
                e = s + nlayer
            # then add the time dimension on top of that
            da = xr.concat(das_layer, dim="time")
        else:
            das.sort(key=lambda da: da.coords["layer"])
            da = xr.concat(das, dim="layer")
    else:
        if hastime:
            das.sort(key=lambda da: da.coords["time"])
            da = xr.concat(das, dim="time")
        else:
            assert len(das) == 1
            da = das[0]

    return da


def loadset(globpath, chunks=None, memmap=False):
    """Read a set of parameters to a dict of xarray.DataArray
    
    Compared to imod.idf.load, this function lets you read multiple parameters
    at once, which will each be a separate entry in an OrderedDict, with as key
    the parameter name, and as value the xarray.DataArray.

    Parameters
    ----------
    globpath : str or Path
        A glob pattern expansion such as `'model/**/*.idf'`, which recursively
        finds all IDF files under the model directory. Note that files with
        the same name (part before the first underscore) wil be combined into
        a single xarray.DataArray.
    chunks : int or tuple of int, optional
        How to chunk the array. By default it creates only 1 chunk.
    memmap : bool, optional
        Whether to use a memory map to the file, or an in memory
        copy. Default is to use a memory map.
    
    Returns
    -------
    OrderedDict
        OrderedDict of str (parameter name) to xarray.DataArray.
        All metadata needed for writing the file to IDF or other formats
        using imod.rasterio are included in the xarray.DataArray.attrs.
    """
    # convert since for Path.glob non-relative patterns are unsupported
    if isinstance(globpath, Path):
        globpath = str(globpath)

    paths = [Path(p) for p in glob(globpath, recursive=True)]

    n = len(paths)
    if n == 0:
        raise FileNotFoundError("Could not find any files matching {}".format(globpath))
    # group the DataArrays together using their name
    # note that directory names are ignored, and in case of duplicates, the last one wins
    names = [util.decompose(path)["name"] for path in paths]
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
def save(path, a, nodata=1.e20):
    """Write a xarray.DataArray to one or more IDF files

    If the DataArray only has `y` and `x` dimensions, a single IDF file is
    written, like the `imod.idf.write` function. This function is more general
    and also supports `time` and `layer` dimensions. It will split these up,
    give them their own filename according to the conventions in
    `imod.util.compose`, and write them each

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be written. This function decides on the
        actual filename(s) using conventions, so it only takes the directory and
        name from this parameter. 
    a : xarray.DataArray
        DataArray to be written. It needs to have exactly a.dims == ('y', 'x').

    Example
    -------
    Consider a DataArray `da` that has dimensions 'layer', 'y' and 'x', with the
    'layer' dimension consisting of layer 1 and 2::

        save('path/to/head', da)

    This writes the following two IDF files: 'path/to/head_l1.idf' and
    'path/to/head_l2.idf'.
    """
    d = util.decompose(path)
    d["extension"] = ".idf"
    d["directory"].mkdir(exist_ok=True, parents=True)

    # handle the case where they are not a dim but are a coord
    # i.e. you only have one layer but you did a.assign_coords(layer=1)
    # in this case we do want _l1 in the IDF file name
    check_coords = ["layer", "time"]
    for coord in check_coords:
        if (coord in a.coords) and not (coord in a.dims):
            if coord == "time":
                # .item() gives an integer, we need a pd.Timestamp or datetime.datetime
                dt64 = a.coords[coord].values
                val = pd.Timestamp(dt64)
            else:
                val = a.coords[coord].item()
            d[coord] = val

    # stack all non idf dims into one new idf dimension,
    # over which we can then iterate to write all individual idfs
    extradims = _extra_dims(a)
    if extradims:
        stacked = a.stack(idf=extradims)
        for coordvals, a_yx in list(stacked.groupby("idf")):
            # set the right layer/timestep/etc in the dict to make the filename
            d.update(dict(zip(extradims, coordvals)))
            fn = util.compose(d)
            write(fn, a_yx)
    else:
        # no extra dims, only one IDF
        fn = util.compose(d)
        write(fn, a)


def _extra_dims(a):
    dims = filter(lambda dim: dim not in ("y", "x"), a.dims)
    return list(dims)


def write(path, a, nodata=1.e20):
    """Write a 2D xarray.DataArray to a IDF file

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be written
    a : xarray.DataArray
        DataArray to be written. It needs to have exactly a.dims == ('y', 'x').
    """
    assert a.dims == ("y", "x")
    with open(path, "wb") as f:
        f.write(pack("i", 1271))  # Lahey RecordLength Ident.
        nrow = a.y.size
        ncol = a.x.size
        attrs = a.attrs
        itb = isinstance(attrs.get("top", None), (int, float)) and isinstance(
            attrs.get("bot", None), (int, float)
        )
        f.write(pack("i", ncol))
        f.write(pack("i", nrow))
        # IDF supports only incrementing x, and decrementing y
        dx, xmin, xmax, dy, ymin, ymax = util.spatial_reference(a)
        if dy > 0.0:
            a.values = np.flip(a.values, axis=0)
        if dx < 0.0:
            a.values = np.flip(a.values, axis=1)
        dx = abs(dx)
        dy = abs(dy)

        f.write(pack("f", xmin))
        f.write(pack("f", xmax))
        f.write(pack("f", ymin))
        f.write(pack("f", ymax))
        f.write(pack("f", float(a.min())))  # dmin
        f.write(pack("f", float(a.max())))  # dmax
        f.write(pack("f", nodata))
        f.write(pack("?", False))  # ieq
        f.write(pack("?", itb))
        f.write(pack("?", False))  # ivf
        f.write(pack("x"))  # not used
        f.write(pack("f", dx))
        f.write(pack("f", dy))
        if itb:
            f.write(pack("f", attrs["top"]))
            f.write(pack("f", attrs["bot"]))
        # convert to a numpy.ndarray of float32
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        a = a.fillna(nodata)
        a.values.tofile(f)
