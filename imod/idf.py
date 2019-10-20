"""
Functions for reading and writing iMOD Data Files (IDFs) to ``xarray`` objects.

The primary functions to use are :func:`imod.idf.open` and
:func:`imod.idf.save`, though lower level functions are also available.
"""

import collections
import functools
import glob
import itertools
import pathlib
import re
import struct

import dask
import numpy as np
import xarray as xr

from imod import util
from . import array_IO

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
    return array_IO.reading._to_nan(a, nodata)


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
        to the examples for ``imod.idf.open``.

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


# Open IDFs for multiple times and/or layers into one DataArray
def open(path, use_cftime=False, pattern=None):
    r"""
    Open one or more IDF files as an xarray.DataArray.

    In accordance with xarray's design, ``open`` loads the data of IDF files
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
        Use ``cftime.DatetimeProlepticGregorian`` instead of `np.datetime64[ns]`
        for the time axis.

        Dates are normally encoded as ``np.datetime64[ns]``; however, if dates
        fall before 1678 or after 2261, they are automatically encoded as
        ``cftime.DatetimeProlepticGregorian`` objects rather than
        ``np.datetime64[ns]``.
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

    The same, this time explicitly specifying ``name``, ``time``, and ``layer``:

    >>> head = imod.idf.open("head_2001*_l*.idf", pattern="{name}_{time}_l{layer}")

    The format string pattern will only work on tidy paths, where variables are
    separated by underscores. You can also pass a compiled regex pattern.
    Make sure to include the ``re.IGNORECASE`` flag since all paths are lowered.

    >>> import re
    >>> pattern = re.compile(r"(?P<name>[\w]+)L(?P<layer>[\d+]*)", re.IGNORECASE)
    >>> head = imod.idf.open("headL11", pattern=pattern)

    However, this requires constructing regular expressions, which is
    generally a fiddly process. Regex notation is also impossible to
    remember. The website https://regex101.com is a nice help. Alternatively,
    the most pragmatic solution may be to just rename your files.
    """

    if isinstance(path, list):
        return array_IO.reading._load(path, use_cftime, pattern, _read, header)
    elif isinstance(path, pathlib.Path):
        path = str(path)

    paths = [pathlib.Path(p) for p in glob.glob(path)]
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"Could not find any files matching {path}")
    return array_IO.reading._load(paths, use_cftime, pattern, _read, header)


def _merge_subdomains(pathlists, use_cftime, pattern):
    das = [
        array_IO.reading._load(pathlist, use_cftime, pattern, _read, header)
        for pathlist in pathlists.values()
    ]
    x = np.unique(np.concatenate([da.x.values for da in das]))
    y = np.unique(np.concatenate([da.y.values for da in das]))

    nrow = y.size
    ncol = x.size
    nlayer = das[0].coords["layer"].size
    out = np.full((1, nlayer, nrow, ncol), np.nan)

    for da in das:
        ix = np.searchsorted(x, da.x.values[0], side="left")
        iy = nrow - np.searchsorted(y, da.y.values[0], side="right")
        _, _, ysize, xsize = da.shape
        out[:, :, iy : iy + ysize, ix : ix + xsize] = da.values

    return out


def open_subdomains(path, use_cftime=False):
    """
    Combine IDF files of multiple subdomains.

    Parameters
    ----------
    path : str, Path or list
    use_cftime : bool, optional
    pattern : str, regex pattern, optional

    Returns
    -------
    xarray.DataArray

    """
    if isinstance(path, pathlib.Path):
        path = str(path)
    paths = glob.glob(path)
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"Could not find any files matching {path}")

    pattern = re.compile(
        r"[\w-]+_(?P<time>[0-9-]+)_l(?P<layer>[0-9]+)_p(?P<subdomain>[0-9]{3})",
        re.IGNORECASE,
    )
    # There are no real benefits to itertools.groupby in this case, as there's
    # no benefit to using a (lazy) iterator in this case
    grouped_by_time = collections.defaultdict(
        functools.partial(collections.defaultdict, list)
    )
    count_per_subdomain = collections.defaultdict(int)  # used only for checking counts
    timestrings = []
    layers = []
    numbers = []
    for p in paths:
        search = pattern.search(p)
        timestr = search.group(1)
        layer = int(search.group(2))
        number = int(search.group(3))
        grouped_by_time[timestr][number].append(p)
        count_per_subdomain[number] += 1
        numbers.append(number)
        layers.append(layer)
        timestrings.append(timestr)

    # Test whether subdomains are complete
    numbers = sorted(set(numbers))
    first = numbers[0]
    first_len = count_per_subdomain[first]
    for number in numbers:
        group_len = count_per_subdomain[number]
        if group_len != first_len:
            raise ValueError(
                f"The number of IDFs are not identical for every subdomain. "
                f"Subdomain p{first} has {first_len} IDF files, subdomain p{number} "
                f"has {group_len} IDF files."
            )

    pattern = r"{name}_{time}_l{layer}_p\d+"
    timestrings = sorted(set(timestrings))

    # Prepare output coordinates
    coords = {}
    first_time = timestrings[0]
    samplingpaths = [
        pathlist[first] for pathlist in grouped_by_time[first_time].values()
    ]
    headers = [header(path, pattern) for path in samplingpaths]
    subdomain_bounds = [(h["xmin"], h["xmax"], h["ymin"], h["ymax"]) for h in headers]
    subdomain_cellsizes = [(h["dx"], h["dy"]) for h in headers]
    subdomain_coords = [
        util._xycoords(bounds, cellsizes)
        for bounds, cellsizes in zip(subdomain_bounds, subdomain_cellsizes)
    ]
    coords["y"] = np.unique(
        np.concatenate([coords["y"] for coords in subdomain_coords])
    )[::-1]
    coords["x"] = np.unique(
        np.concatenate([coords["x"] for coords in subdomain_coords])
    )
    coords["layer"] = np.array(sorted(set(layers)))
    times = [util.to_datetime(timestr) for timestr in timestrings]
    times, use_cftime = util._convert_datetimes(times, use_cftime)
    if use_cftime:
        coords["time"] = xr.CFTimeIndex(np.unique(times))
    else:
        coords["time"] = np.unique(times)
    shape = (1, coords["layer"].size, coords["y"].size, coords["x"].size)
    dims = ("time", "layer", "y", "x")

    # Collect and merge data
    merged = []
    for group in grouped_by_time.values():
        # Build a single array per timestep
        timestep_data = dask.delayed(_merge_subdomains)(group, use_cftime, pattern)
        dask_array = dask.array.from_delayed(timestep_data, shape, dtype=np.float32)
        merged.append(dask_array)
    data = dask.array.concatenate(merged, axis=0)

    # Get tops and bottoms if possible
    headers = [header(path, pattern) for path in grouped_by_time[first_time][first]]
    tops = [c.get("top", None) for c in headers]
    bots = [c.get("bot", None) for c in headers]
    layers = [c.get("layer", None) for c in headers]
    _, unique_indices = np.unique(layers, return_index=True)
    all_have_z = all(map(lambda v: v is not None, itertools.chain(tops, bots)))
    if all_have_z:
        if coords["layer"].size > 1:
            coords = _array_z_coord(coords, tops, bots, unique_indices)
        else:
            coords = _scalar_z_coord(coords, tops, bots)

    return xr.DataArray(data, coords, dims)


def open_dataset(globpath, use_cftime=False, pattern=None):
    """
    Open a set of IDFs to a dict of xarray.DataArrays.

    Compared to imod.idf.open, this function lets you open multiple parameters
    at once (for example kh values and starting heads of a model), which will
    each be a separate entry in a dictionary, with as key the parameter name,
    and as value the xarray.DataArray.

    Parameters
    ----------
    globpath : str or Path
        A glob pattern expansion such as ``'model/**/*.idf'``, which recursively
        finds all IDF files under the model directory. Note that files with
        the same name (part before the first underscore) wil be combined into
        a single xarray.DataArray.
    use_cftime : bool, optional
        Use ``cftime.DatetimeProlepticGregorian`` instead of `np.datetime64[ns]`
        for the time axis.

        Dates are normally encoded as ``np.datetime64[ns]``; however, if dates
        fall before 1679 or after 2262, they are automatically encoded as
        ``cftime.DatetimeProlepticGregorian`` objects rather than
        ``np.datetime64[ns]``.
    pattern : str, regex pattern, optional
        If the filenames do match default naming conventions of
        {name}_{time}_l{layer}, a custom pattern can be defined here either
        as a string, or as a compiled regular expression pattern. Please refer
        to the examples for ``imod.idf.open``.

    Returns
    -------
    collections.OrderedDict
        Dictionary of str (parameter name) to xarray.DataArray.
        All metadata needed for writing the file to IDF or other formats
        using imod.rasterio are included in the xarray.DataArray.attrs.
    """

    # convert since for Path.glob non-relative patterns are unsupported
    if isinstance(globpath, pathlib.Path):
        globpath = str(globpath)

    paths = [pathlib.Path(p) for p in glob.glob(globpath, recursive=True)]

    n = len(paths)
    if n == 0:
        raise FileNotFoundError("Could not find any files matching {}".format(globpath))
    # group the DataArrays together using their name
    # note that directory names are ignored, and in case of duplicates, the last one wins
    names = [util.decompose(path, pattern)["name"] for path in paths]
    unique_names = list(np.unique(names))
    d = collections.OrderedDict()
    for n in unique_names:
        d[n] = []  # prepare empty lists to append to
    for p, n in zip(paths, names):
        d[n].append(p)

    # load each group into a DataArray
    das = [
        array_IO.reading._load(v, use_cftime, pattern, _read, header)
        for v in d.values()
    ]

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

        itb = False
        if "z" in a.coords and "dz" in a.coords:
            z = a.coords["z"]
            dz = abs(a.coords["dz"])
            try:
                top = float(z + 0.5 * dz)
                bot = float(z - 0.5 * dz)
                itb = True
            except TypeError:  # not a scalar value
                pass

        f.write(struct.pack("?", itb))
        f.write(struct.pack("xx"))  # not used
        if ieq:
            f.write(struct.pack("f", dx))
            f.write(struct.pack("f", -dy))
        if itb:
            f.write(struct.pack("f", top))
            f.write(struct.pack("f", bot))
        if not ieq:
            a.coords["dx"].values.astype(np.float32).tofile(f)
            (-a.coords["dy"].values).astype(np.float32).tofile(f)
        # convert to a numpy.ndarray of float32
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        a = a.fillna(nodata)
        a.values.tofile(f)


def _as_voxeldata(a):
    """
    If "z" is present as a dimension, generate layer if necessary. Ensure that
    layer is the dimension (via swap_dims). Infer "dz" if necessary, and if
    possible.

    Parameters
    ----------
    a : xr.DataArray

    Returns
    -------
    a : xr.DataArray
        copy of input a, with swapped dims and dz added, if necessary.
    """
    # Avoid side-effects
    a = a.copy()

    if "z" in a.coords:
        if "z" in a.dims:  # it's definitely 1D
            # have to swap it with layer in this case
            if "layer" not in a.coords:
                a = a.assign_coords(layer=("z", np.arange(1, a["z"].size + 1)))
            a = a.swap_dims({"z": "layer"})

            # It'll raise an Error if it cannot infer dz
            if "dz" not in a.coords:
                dz, _, _ = util.coord_reference(a["z"])
                if isinstance(dz, float):
                    a = a.assign_coords(dz=dz)
                else:
                    a = a.assign_coords(dz=("layer", dz))

        elif len(a["z"].shape) == 1:  # one dimensional
            if "layer" in a.coords:
                # Check if z depends only on layer
                if tuple(a["z"].indexes.keys()) == ("layer",):
                    if "dz" not in a.coords:
                        # It'll raise an Error if it cannot infer dz
                        dz, _, _ = util.coord_reference(a["z"])
                        if isinstance(dz, float):
                            a = a.assign_coords(dz=dz)
                        else:
                            a = a.assign_coords(dz=("layer", dz))
    return a


def save(path, a, nodata=1.0e20, pattern=None):
    """
    Write a xarray.DataArray to one or more IDF files

    If the DataArray only has ``y`` and ``x`` dimensions, a single IDF file is
    written, like the ``imod.idf.write`` function. This function is more general
    and also supports ``time`` and ``layer`` dimensions. It will split these up,
    give them their own filename according to the conventions in
    ``imod.util.compose``, and write them each.

    Parameters
    ----------
    path : str or Path
        Path to the IDF file to be written. This function decides on the
        actual filename(s) using conventions.
    a : xarray.DataArray
        DataArray to be written. It needs to have dimensions ('y', 'x'), and
        optionally ``layer`` and ``time``.
    nodata : float, optional
        Nodata value in the saved IDF files. Xarray uses nan values to represent
        nodata, but these tend to work unreliably in iMOD(FLOW).
        Defaults to a value of 1.0e20.
    pattern : str
        Format string which defines how to create the filenames. See examples.

    Example
    -------
    Consider a DataArray ``da`` that has dimensions 'layer', 'y' and 'x', with the
    'layer' dimension consisting of layer 1 and 2::

        save('path/to/head', da)

    This writes the following two IDF files: 'path/to/head_l1.idf' and
    'path/to/head_l2.idf'.


    It is possible to generate custom filenames using a format string. The
    default filenames would be generated by the following format string:

        save("example", pattern="{name}_l{layer}{extension}")

    If you desire zero-padded numbers that show up neatly sorted in a
    file manager, you may specify:

        save("example", pattern="{name}_l{layer:02d}{extension}")

    In this case, a 0 will be padded for single digit numbers ('1' will become
    '01').

    To get a date with dashes, use the following pattern:

        "{name}_{time:%Y-%m-%d}_l{layer}{extension}"

    """
    # Swap coordinates if possible, add "dz" if possible.
    a = _as_voxeldata(a)

    # Deal with path
    path = pathlib.Path(path)

    if path.suffix == "":
        path = path.with_suffix(".idf")

    array_IO.writing._save(path, a, nodata, pattern, write)
