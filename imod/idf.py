import collections
import glob
import itertools
import pathlib
import struct
import warnings

import dask
import numpy as np
import pandas as pd
import xarray as xr

import imod
from imod import util

# Make sure we can still use the built-in function...
f_open = open


def header(path, pattern):
    """Read the IDF header information into a dictionary"""
    attrs = util.decompose(path, pattern)
    with f_open(path, "rb") as f:
        reclen_id = struct.unpack("i", f.read(4))[0]  # Lahey RecordLength Ident.
        if reclen_id != 1271:
            raise ValueError(f"Not a supported IDF file: {path}")
        ncol = struct.unpack("i", f.read(4))[0]
        nrow = struct.unpack("i", f.read(4))[0]
        attrs["xmin"] = struct.unpack("f", f.read(4))[0]
        attrs["xmax"] = struct.unpack("f", f.read(4))[0]
        attrs["ymin"] = struct.unpack("f", f.read(4))[0]
        attrs["ymax"] = struct.unpack("f", f.read(4))[0]
        # dmin and dmax are recomputed during writing
        f.read(4)  # dmin, minimum data value present
        f.read(4)  # dmax, maximum data value present
        nodata = struct.unpack("f", f.read(4))[0]
        attrs["nodata"] = nodata
        # flip definition here such that True means equidistant
        # equidistant IDFs
        ieq = not struct.unpack("?", f.read(1))[0]
        itb = struct.unpack("?", f.read(1))[0]
        f.read(2)  # not used

        if ieq:
            # dx and dy are stored positively in the IDF
            # dy is made negative here to be consistent with the nonequidistant case
            attrs["dx"] = struct.unpack("f", f.read(4))[0]
            attrs["dy"] = -struct.unpack("f", f.read(4))[0]

        if itb:
            attrs["top"] = struct.unpack("f", f.read(4))[0]
            attrs["bot"] = struct.unpack("f", f.read(4))[0]

        if not ieq:
            # dx and dy are stored positive in the IDF, but since the difference between
            # successive y coordinates is negative, it is made negative here
            attrs["dx"] = np.fromfile(f, np.float32, ncol)
            attrs["dy"] = -np.fromfile(f, np.float32, nrow)

        # These are derived, remove after using them downstream
        attrs["headersize"] = f.tell()
        attrs["ncol"] = ncol
        attrs["nrow"] = nrow

    return attrs


def _all_equal(seq, elem):
    """Raise an error if not all elements of a list are equal"""
    if not seq.count(seq[0]) == len(seq):
        raise ValueError(f"All {elem} must be the same, found: {set(seq)}")


def _check_cellsizes(cellsizes):
    """
    Checks if cellsizes match, raises ValueError otherwise

    Parameters
    ----------
    cellsizes : list of tuples
        tuples may contain:
        * two floats
        * two ndarrays

    Returns
    -------
    None
    """
    msg = "Cellsizes of IDFs do not match"
    try:
        if not (cellsizes.count(cellsizes[0]) == len(cellsizes)):
            raise ValueError(msg)
    except ValueError:  # contains ndarrays
        try:
            # all ndarrays
            if not all(np.allclose(cellsizes[0], c) for c in cellsizes):
                raise ValueError(msg)
        except ValueError:
            # some ndarrays, some floats
            # create floats for comparison with allclose
            try:
                dx = cellsizes[0][0][0]
                dy = cellsizes[0][1][0]
            except TypeError:
                dx = cellsizes[0][0]
                dy = cellsizes[0][1]
            # comparison
            for cellsize in cellsizes:
                # Unfortunately this allocates by broadcasting dx and dy
                if not np.allclose(cellsize[0], dx):
                    raise ValueError(msg)
                if not np.allclose(cellsize[1], dy):
                    raise ValueError(msg)


def _sort_time_layer(header):
    """Key to sort headers by time and layer

    Works regardless of whether time and layer is present in the header,
    by returning the sortable but constant 0 if it is not. Not that this
    requires that all headers in the list either have or don't have time
    or layer. This is ensured by _all_or_nothing.
    """
    time = header.get("time", 0)
    layer = header.get("layer", 0)
    return time, layer


def _has_dim(seq):
    """Check if either 0 or all None are present

    Returns
    -------
    True if no None in seq, False if all None, error otherwise
    """
    nones = [x is None for x in seq]
    if any(nones):
        if all(nones):
            return False
        else:
            raise ValueError("Either 0 or all None allowed")
    return True


def _xycoords(bounds, cellsizes):
    """Based on bounds and cellsizes, construct coords with spatial information"""
    # unpack tuples
    xmin, xmax, ymin, ymax = bounds
    dx, dy = cellsizes
    coords = collections.OrderedDict()
    # from cell size to x and y coordinates
    if isinstance(dx, (int, float)):  # equidistant
        coords["x"] = np.arange(xmin + dx / 2.0, xmax, dx)
        coords["y"] = np.arange(ymax + dy / 2.0, ymin, dy)
        coords["dx"] = float(dx)
        coords["dy"] = float(dy)
    else:  # nonequidistant
        # even though IDF may store them as float32, we always convert them to float64
        dx = dx.astype(np.float64)
        dy = dy.astype(np.float64)
        coords["x"] = xmin + np.cumsum(dx) - 0.5 * dx
        coords["y"] = ymax + np.cumsum(dy) - 0.5 * dy
        coords["dx"] = ("x", dx)
        coords["dy"] = ("y", dy)
    return coords


def _to_nan(a, nodata):
    """Change all nodata values in the array to NaN"""
    # it needs to be NaN for xarray to deal with it properly
    # no need to store the nodata value if it is always NaN
    if np.isnan(nodata):
        return a
    else:
        isnodata = np.isclose(a, nodata)
        a[isnodata] = np.nan
        return a


def _read(path, headersize, nrow, ncol, nodata):
    """
    Read a single IDF file to a numpy.ndarray

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be read
    headersize : int
        byte size of header
    nrow : int
    ncol : int
    nodata : np.float32

    Returns
    -------
    numpy.ndarray
        A float32 numpy.ndarray with shape (nrow, ncol) of the values
        in the IDF file. On opening all nodata values are changed
        to NaN in the numpy.ndarray.
    """
    with f_open(path, "rb") as f:
        f.seek(headersize)
        a = np.reshape(np.fromfile(f, np.float32, nrow * ncol), (nrow, ncol))
    return _to_nan(a, nodata)


def read(path, pattern=None):
    """
    Read a single IDF file to a numpy.ndarray

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be read
    pattern : str, regex pattern, optional
        If the filenames do match default naming conventions of
        {name}_{time}_l{layer}, a custom pattern can be defined here either
        as a string, or as a compiled regular expression pattern. Please refer
        to the examples for `imod.idf.open`.

    Returns
    -------
    numpy.ndarray
        A float32 numpy.ndarray with shape (nrow, ncol) of the values
        in the IDF file. On opening all nodata values are changed
        to NaN in the numpy.ndarray.
    dict
        A dict with all metadata.
    """

    attrs = header(path, pattern)
    headersize = attrs.pop("headersize")
    nrow = attrs.pop("nrow")
    ncol = attrs.pop("ncol")
    nodata = attrs.pop("nodata")
    return _read(path, headersize, nrow, ncol, nodata), attrs


def _dask(path, memmap=False, attrs=None, pattern=None):
    """
    Read a single IDF file to a dask.array

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be read
    attrs : dict, optional
        A dict as returned by imod.idf.header, this function is called if not supplied.
        Used to minimize unneeded filesystem calls.
    pattern : str, regex pattern, optional
        If the filenames do match default naming conventions of
        {name}_{time}_l{layer}, a custom pattern can be defined here either
        as a string, or as a compiled regular expression pattern. Please refer
        to the examples in `imod.idf.open`.

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
        warnings.warn("memmap option is removed", FutureWarning)

    if attrs is None:
        attrs = header(path, pattern)
    # If we don't unpack, it seems we run into trouble with the dask array later
    # on, probably because attrs isn't immutable. This works fine instead.
    headersize = attrs.pop("headersize")
    nrow = attrs["nrow"]
    ncol = attrs["ncol"]
    nodata = attrs.pop("nodata")
    # dask.delayed requires currying
    a = dask.delayed(imod.idf._read)(path, headersize, nrow, ncol, nodata)
    x = dask.array.from_delayed(a, shape=(nrow, ncol), dtype=np.float32)
    return x, attrs


def dataarray(path, memmap=False):
    """
    Read a single IDF file to a xarray.DataArray

    The function imod.idf.open is more general and can load multiple layers
    and/or timestamps at once.

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be read
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
    warnings.warn(
        "imod.idf.dataarray is deprecated, use imod.idf.open instead", FutureWarning
    )
    return _load([path], False)


def load(path, memmap=False, use_cftime=False):
    """
    load is deprecated. Check the documentation for `imod.idf.open` instead.
    """
    warnings.warn("load is deprecated, use imod.idf.open instead.", FutureWarning)
    return open(path, memmap, use_cftime)


# Open IDFs for multiple times and/or layers into one DataArray
def open(path, memmap=False, use_cftime=False, pattern=None):
    r"""
    Open one or more IDF files as an xarray.DataArray.

    In accordance with xarray's design, `open` loads the data of IDF files
    lazily. This means the data of the IDFs are not loaded into memory until the
    data is needed. This allows for easier handling of large datasets, and
    more efficient computations.

    Parameters
    ----------
    path : str, Path or list
        This can be a single file, 'head_l1.idf', a glob pattern expansion,
        'head_l*.idf', or a list of files, ['head_l1.idf', 'head_l2.idf'].
        Note that each file needs to be of the same name (part before the
        first underscore) but have a different layer and/or timestamp,
        such that they can be combined in a single xarray.DataArray.
    use_cftime : bool, optional
        Use `cftime.DatetimeProlepticGregorian` instead of `np.datetime64[ns]`
        for the time axis.

        Dates are normally encoded as `np.datetime64[ns]`; however, if dates
        fall before 1678 or after 2261, they are automatically encoded as
        `cftime.DatetimeProlepticGregorian` objects rather than
        `np.datetime64[ns]`.
    pattern : str, regex pattern, optional
        If the filenames do match default naming conventions of
        {name}_{time}_l{layer}, a custom pattern can be defined here either
        as a string, or as a compiled regular expression pattern. See the
        examples below.

    Returns
    -------
    xarray.DataArray
        A float32 xarray.DataArray of the values in the IDF file(s).
        All metadata needed for writing the file to IDF or other formats
        using imod.rasterio are included in the xarray.DataArray.attrs.

    Examples
    --------
    Open an IDF file:
    >>> da = imod.idf.open("example.idf")

    Open an IDF file, relying on default naming conventions to identify
    layer:
    >>> da = imod.idf.open("example_l1.idf")

    Open an IDF file, relying on default naming conventions to identify layer
    and time:
    >>> head = imod.idf.open("head_20010101_l1.idf")

    Open multiple IDF files, in this case files for the year 2001 for all
    layers, again relying on default conventions for naming:

    >>> head = imod.idf.open("head_2001*_l*.idf")

    The same, this time explicitly specifying `name`, `time`, and `layer`:

    >>> head = imod.idf.open("head_2001*_l*.idf", pattern="{name}_{time}_l{layer}")

    The format string pattern will only work on tidy paths, where variables are
    separated by underscores. You can also pass a compiled regex pattern.
    Make sure to include the `re.IGNORECASE` flag since all paths are lowered.

    >>> import re
    >>> pattern = re.compile(r"(?P<name>[\w]+)L(?P<layer>[\d+]*)", re.IGNORECASE)
    >>> head = imod.idf.open("headL11", pattern=pattern)

    However, this requires constructing regular expressions, which is
    generally a fiddly process. Regex notation is also impossible to
    remember. The website https://regex101.com is a nice help. Alternatively,
    the most pragmatic solution may be to just rename your files.
    """
    if memmap:
        warnings.warn("memmap option is removed", FutureWarning)

    if isinstance(path, list):
        return _load(path, use_cftime)
    elif isinstance(path, pathlib.Path):
        path = str(path)

    paths = [pathlib.Path(p) for p in glob.glob(path)]
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"Could not find any files matching {path}")
    return _load(paths, use_cftime, pattern)


def _load(paths, use_cftime, pattern):
    """Combine a list of paths to IDFs to a single xarray.DataArray"""
    # this function also works for single IDFs

    headers_unsorted = [imod.idf.header(p, pattern) for p in paths]
    names_unsorted = [h["name"] for h in headers_unsorted]
    _all_equal(names_unsorted, "names")

    # sort headers and paths by time then layer
    zipped = zip(headers_unsorted, paths)
    zipped_sorted = sorted(zipped, key=lambda pair: _sort_time_layer(pair[0]))
    headers, paths = map(list, zip(*zipped_sorted))

    times = [c.get("time", None) for c in headers]
    layers = [c.get("layer", None) for c in headers]
    bounds = [(h["xmin"], h["xmax"], h["ymin"], h["ymax"]) for h in headers]
    cellsizes = [(h["dx"], h["dy"]) for h in headers]

    hastime = _has_dim(times)
    haslayer = _has_dim(layers)
    _all_equal(bounds, "bounding boxes")
    _check_cellsizes(cellsizes)

    # create coordinates
    coords = _xycoords(bounds[0], cellsizes[0])
    dims = ["y", "x"]
    # order matters here due to inserting dims
    if haslayer:
        coords["layer"] = np.unique(layers)
        dims.insert(0, "layer")
    if hastime:
        times, use_cftime = util._convert_datetimes(times, use_cftime)
        if use_cftime:
            coords["time"] = xr.CFTimeIndex(np.unique(times))
        else:
            coords["time"] = np.unique(times)
        dims.insert(0, "time")

    # avoid calling imod.idf.header again here with attrs keyword
    dask_arrays = [
        imod.idf._dask(path, attrs=attrs)[0] for (path, attrs) in zip(paths, headers)
    ]

    if hastime and haslayer:
        # first stack layers per timestep, then stack that
        # this order has to match the dims
        dask_timesteps = []
        # for this groupby the argument needs to be presorted, which is done above
        for _, g in itertools.groupby(zip(dask_arrays, times), key=lambda z: z[1]):
            # a list of dask arrays belonging to a single timestep (all layers)
            dask_list = [t[0] for t in g]
            # stack with dask, adding a new dimension to the front
            dask_timestep = dask.array.stack(dask_list, axis=0)
            dask_timesteps.append(dask_timestep)

        dask_array = dask.array.stack(dask_timesteps, axis=0)
    elif hastime or haslayer:
        dask_array = dask.array.stack(dask_arrays, axis=0)
    else:
        dask_array = dask_arrays[0]

    return xr.DataArray(dask_array, coords, dims, name=names_unsorted[0])


def open_dataset(globpath, memmap=False, use_cftime=False, pattern=None):
    """
    Open a set of IDFs to a dict of xarray.DataArrays.

    Compared to imod.idf.open, this function lets you open multiple parameters
    at once (for example kh values and starting heads of a model), which will
    each be a separate entry in a dictionary, with as key the parameter name,
    and as value the xarray.DataArray.

    Parameters
    ----------
    globpath : str or Path
        A glob pattern expansion such as `'model/**/*.idf'`, which recursively
        finds all IDF files under the model directory. Note that files with
        the same name (part before the first underscore) wil be combined into
        a single xarray.DataArray.
    use_cftime : bool, optional
        Use `cftime.DatetimeProlepticGregorian` instead of `np.datetime64[ns]`
        for the time axis.

        Dates are normally encoded as `np.datetime64[ns]`; however, if dates
        fall before 1679 or after 2262, they are automatically encoded as
        `cftime.DatetimeProlepticGregorian` objects rather than
        `np.datetime64[ns]`.
    pattern : str, regex pattern, optional
        If the filenames do match default naming conventions of
        {name}_{time}_l{layer}, a custom pattern can be defined here either
        as a string, or as a compiled regular expression pattern. Please refer
        to the examples for `imod.idf.open`.

    Returns
    -------
    collections.OrderedDict
        Dictionary of str (parameter name) to xarray.DataArray.
        All metadata needed for writing the file to IDF or other formats
        using imod.rasterio are included in the xarray.DataArray.attrs.
    """
    if memmap:
        warnings.warn("memmap option is removed", FutureWarning)

    # convert since for Path.glob non-relative patterns are unsupported
    if isinstance(globpath, pathlib.Path):
        globpath = str(globpath)

    paths = [pathlib.Path(p) for p in glob.glob(globpath, recursive=True)]

    n = len(paths)
    if n == 0:
        raise FileNotFoundError("Could not find any files matching {}".format(globpath))
    # group the DataArrays together using their name
    # note that directory names are ignored, and in case of duplicates, the last one wins
    names = [util.decompose(path)["name"] for path in paths]
    unique_names = list(np.unique(names))
    d = collections.OrderedDict()
    for n in unique_names:
        d[n] = []  # prepare empty lists to append to
    for p, n in zip(paths, names):
        d[n].append(p)

    # load each group into a DataArray
    das = [_load(v, use_cftime, pattern) for v in d.values()]

    # store each DataArray under it's own name in a dictionary
    dd = collections.OrderedDict()
    for da in das:
        dd[da.name] = da
    # Initially I wanted to return a xarray Dataset here,
    # but then realised that it is not always aligned, and therefore not possible, see
    # https://github.com/pydata/xarray/issues/1471#issuecomment-313719395
    # It is not aligned when some parameters only have a non empty subset of a dimension,
    # such as L2 + L3. This dict provides a similar interface anyway. If a Dataset is constructed
    # from unaligned DataArrays it will make copies of the data, which we don't want.
    return dd


def _top_bot_dicts(a):
    """Returns a dictionary with the top and bottom per layer"""
    top = np.atleast_1d(a.attrs["top"]).astype(np.float64)
    bot = np.atleast_1d(a.attrs["bot"]).astype(np.float64)
    if not top.shape == bot.shape:
        raise ValueError('"top" and "bot" attrs should have the same shape')
    if "layer" in a.coords:
        layers = np.atleast_1d(a.coords["layer"].values)
        if not top.shape == layers.shape:
            raise ValueError('"top" attr shape does not match shape of layer dimension')
        d_top = {laynum: t for laynum, t in zip(layers, top)}
        d_bot = {laynum: b for laynum, b in zip(layers, bot)}
    else:
        if not top.shape == (1,):
            raise ValueError(
                'if "layer" is not a coordinate, "top"'
                ' and "bot" attrs should hold only one value'
            )
        d_top = {"no_layer": top[0]}
        d_bot = {"no_layer": bot[0]}
    return d_top, d_bot


# write DataArrays to IDF
def save(path, a, nodata=1.0e20):
    """
    Write a xarray.DataArray to one or more IDF files

    If the DataArray only has `y` and `x` dimensions, a single IDF file is
    written, like the `imod.idf.write` function. This function is more general
    and also supports `time` and `layer` dimensions. It will split these up,
    give them their own filename according to the conventions in
    `imod.util.compose`, and write them each.

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be written. This function decides on the
        actual filename(s) using conventions, so it only takes the directory and
        name from this parameter.
    a : xarray.DataArray
        DataArray to be written. It needs to have dimensions ('y', 'x'), and 
        optionally `layer` and `time`.
    nodata : float, optional
        Nodata value in the saved IDF files. Xarray uses nan values to represent
        nodata, but these tend to work unreliably in iMOD(FLOW).
        Defaults to a value of 1.0e20.

    Example
    -------
    Consider a DataArray `da` that has dimensions 'layer', 'y' and 'x', with the
    'layer' dimension consisting of layer 1 and 2::

        save('path/to/head', da)

    This writes the following two IDF files: 'path/to/head_l1.idf' and
    'path/to/head_l2.idf'.
    """
    if not isinstance(a, xr.DataArray):
        raise TypeError("Data to save must be an xarray.DataArray")

    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.suffix != "":
        raise ValueError(
            "`imod.idf.save` generates time, layer, and file extension for the path."
            " Use `imod.idf.write` instead to write a single IDF file with a fully"
            " specified path."
        )

    # A more flexible schema might be required to support additional variables
    # such as species, for concentration. The straightforward way is by giving
    # a format string, e.g.: {name}_{time}_l{layer}
    # Find the vars in curly braces, and validate with da.coords
    d = {"extension": ".idf", "name": path.stem, "directory": path.parent}
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

    # Allow tops and bottoms to be written for voxel like IDFs.
    has_topbot = False
    if "top" in a.attrs and "bot" in a.attrs:
        has_topbot = True
        d_top, d_bot = _top_bot_dicts(a)

    # stack all non idf dims into one new idf dimension,
    # over which we can then iterate to write all individual idfs
    extradims = _extra_dims(a)
    if extradims:
        stacked = a.stack(idf=extradims)
        for coordvals, a_yx in list(stacked.groupby("idf")):
            # set the right layer/timestep/etc in the dict to make the filename
            d.update(dict(zip(extradims, coordvals)))
            fn = util.compose(d)
            if has_topbot:
                layer = d.get("layer", "no_layer")
                a_yx.attrs["top"] = d_top[layer]
                a_yx.attrs["bot"] = d_bot[layer]
            write(fn, a_yx, nodata)
    else:
        # no extra dims, only one IDF
        fn = util.compose(d)
        write(fn, a, nodata)


def _extra_dims(a):
    dims = filter(lambda dim: dim not in ("y", "x"), a.dims)
    return list(dims)


def write(path, a, nodata=1.0e20):
    """
    Write a 2D xarray.DataArray to a IDF file

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be written
    a : xarray.DataArray
        DataArray to be written. It needs to have exactly a.dims == ('y', 'x').
    nodata : float, optional
        Nodata value in the saved IDF files. Xarray uses nan values to represent
        nodata, but these tend to work unreliably in iMOD(FLOW).
        Defaults to a value of 1.0e20.

    """
    if not isinstance(a, xr.DataArray):
        raise TypeError("Data to write must be an xarray.DataArray")
    if not a.dims == ("y", "x"):
        raise ValueError("Dimensions must be exactly ('y', 'x').")

    with f_open(path, "wb") as f:
        f.write(struct.pack("i", 1271))  # Lahey RecordLength Ident.
        nrow = a.y.size
        ncol = a.x.size
        attrs = a.attrs
        itb = isinstance(attrs.get("top", None), (int, float)) and isinstance(
            attrs.get("bot", None), (int, float)
        )
        f.write(struct.pack("i", ncol))
        f.write(struct.pack("i", nrow))
        dx, xmin, xmax, dy, ymin, ymax = util.spatial_reference(a)
        # IDF supports only incrementing x, and decrementing y
        if (np.atleast_1d(dx) < 0.0).all():
            raise ValueError("dx must be positive")
        if (np.atleast_1d(dy) > 0.0).all():
            raise ValueError("dy must be negative")

        f.write(struct.pack("f", xmin))
        f.write(struct.pack("f", xmax))
        f.write(struct.pack("f", ymin))
        f.write(struct.pack("f", ymax))
        f.write(struct.pack("f", float(a.min())))  # dmin
        f.write(struct.pack("f", float(a.max())))  # dmax
        f.write(struct.pack("f", nodata))

        if isinstance(dx, float) and isinstance(dy, float):
            ieq = True  # equidistant
            f.write(struct.pack("?", not ieq))  # ieq
        else:
            ieq = False  # nonequidistant
            f.write(struct.pack("?", not ieq))  # ieq

        f.write(struct.pack("?", itb))
        f.write(struct.pack("xx"))  # not used
        if ieq:
            f.write(struct.pack("f", dx))
            f.write(struct.pack("f", -dy))
        if itb:
            f.write(struct.pack("f", attrs["top"]))
            f.write(struct.pack("f", attrs["bot"]))
        if not ieq:
            a.coords["dx"].values.astype(np.float32).tofile(f)
            (-a.coords["dy"].values).astype(np.float32).tofile(f)
        # convert to a numpy.ndarray of float32
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        a = a.fillna(nodata)
        a.values.tofile(f)
