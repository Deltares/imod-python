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
import warnings

import dask
import numpy as np
import xarray as xr

from imod import util

from . import array_io

# Make sure we can still use the built-in function...
f_open = open


def header(path, pattern):
    """Read the IDF header information into a dictionary"""
    attrs = util.decompose(path, pattern)
    with f_open(path, "rb") as f:
        reclen_id = struct.unpack("i", f.read(4))[0]  # Lahey RecordLength Ident.
        if reclen_id == 1271:
            floatsize = intsize = 4
            floatformat = "f"
            intformat = "i"
            dtype = "float32"
            doubleprecision = False
        # 2296 was a typo in the iMOD manual. Keep 2296 around in case some IDFs
        # were written with this identifier to avoid possible incompatibility
        # issues.
        elif reclen_id == 2295 or reclen_id == 2296:
            floatsize = intsize = 8
            floatformat = "d"
            intformat = "q"
            dtype = "float64"
            doubleprecision = True
        else:
            raise ValueError(
                f"Not a supported IDF file: {path}\n"
                "Record length identifier should be 1271 or 2295, "
                f"received {reclen_id} instead."
            )

        # Header is fully doubled in size in case of double precision ...
        # This means integers are also turned into 8 bytes
        # and requires padding with some additional bytes
        if doubleprecision:
            f.read(4)  # not used

        ncol = struct.unpack(intformat, f.read(intsize))[0]
        nrow = struct.unpack(intformat, f.read(intsize))[0]
        attrs["xmin"] = struct.unpack(floatformat, f.read(floatsize))[0]
        attrs["xmax"] = struct.unpack(floatformat, f.read(floatsize))[0]
        attrs["ymin"] = struct.unpack(floatformat, f.read(floatsize))[0]
        attrs["ymax"] = struct.unpack(floatformat, f.read(floatsize))[0]
        # dmin and dmax are recomputed during writing
        f.read(floatsize)  # dmin, minimum data value present
        f.read(floatsize)  # dmax, maximum data value present
        nodata = struct.unpack(floatformat, f.read(floatsize))[0]
        attrs["nodata"] = nodata
        # flip definition here such that True means equidistant
        # equidistant IDFs
        ieq = not struct.unpack("?", f.read(1))[0]
        itb = struct.unpack("?", f.read(1))[0]

        f.read(2)  # not used
        if doubleprecision:
            f.read(4)  # not used

        if ieq:
            # dx and dy are stored positively in the IDF
            # dy is made negative here to be consistent with the nonequidistant case
            attrs["dx"] = struct.unpack(floatformat, f.read(floatsize))[0]
            attrs["dy"] = -struct.unpack(floatformat, f.read(floatsize))[0]

        if itb:
            attrs["top"] = struct.unpack(floatformat, f.read(floatsize))[0]
            attrs["bot"] = struct.unpack(floatformat, f.read(floatsize))[0]

        if not ieq:
            # dx and dy are stored positive in the IDF, but since the difference between
            # successive y coordinates is negative, it is made negative here
            attrs["dx"] = np.fromfile(f, dtype, ncol)
            attrs["dy"] = -np.fromfile(f, dtype, nrow)

        # These are derived, remove after using them downstream
        attrs["headersize"] = f.tell()
        attrs["ncol"] = ncol
        attrs["nrow"] = nrow
        attrs["dtype"] = dtype

    return attrs


def _read(path, headersize, nrow, ncol, nodata, dtype):
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
    nodata : np.float

    Returns
    -------
    numpy.ndarray
        A float numpy.ndarray with shape (nrow, ncol) of the values
        in the IDF file. On opening all nodata values are changed
        to NaN in the numpy.ndarray.
    """
    with f_open(path, "rb") as f:
        f.seek(headersize)
        a = np.reshape(np.fromfile(f, dtype, nrow * ncol), (nrow, ncol))
    return array_io.reading._to_nan(a, nodata)


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
        A float numpy.ndarray with shape (nrow, ncol) of the values
        in the IDF file. On opening all nodata values are changed
        to NaN in the numpy.ndarray.
    dict
        A dict with all metadata.
    """
    warnings.warn(
        "The idf.read() function is deprecated. To get a numpy array of an IDF, "
        "use instead: imod.idf.open(path).values",
        FutureWarning,
    )
    attrs = header(path, pattern)
    headersize = attrs.pop("headersize")
    nrow = attrs.pop("nrow")
    ncol = attrs.pop("ncol")
    nodata = attrs.pop("nodata")
    dtype = attrs.pop("dtype")
    return _read(path, headersize, nrow, ncol, nodata, dtype), attrs


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
        A float xarray.DataArray of the values in the IDF file(s).
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

    To ignore the naming conventions, specify ``pattern="{name}"``. This will
    disable parsing of the filename into xarray coordinates.

    >>> head = imod.idf.open("head_20010101_l1.idf", pattern="{name}")

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
        return array_io.reading._load(path, use_cftime, pattern, _read, header)
    elif isinstance(path, pathlib.Path):
        path = str(path)

    paths = [pathlib.Path(p) for p in glob.glob(path)]
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"Could not find any files matching {path}")
    return array_io.reading._load(paths, use_cftime, pattern, _read, header)


def _merge_subdomains(pathlists, use_cftime, pattern):
    das = [
        array_io.reading._load(pathlist, use_cftime, pattern, _read, header)
        for pathlist in pathlists.values()
    ]
    x = np.unique(np.concatenate([da.x.values for da in das]))
    y = np.unique(np.concatenate([da.y.values for da in das]))

    nrow = y.size
    ncol = x.size
    nlayer = das[0].coords["layer"].size
    if "species" in das[0].dims:
        has_species = True
        nspecies = das[0].coords["species"].size
        out = np.full((nspecies, 1, nlayer, nrow, ncol), np.nan)
    else:
        has_species = False
        out = np.full((1, nlayer, nrow, ncol), np.nan)

    for da in das:
        ix = np.searchsorted(x, da.x.values[0], side="left")
        iy = nrow - np.searchsorted(y, da.y.values[0], side="right")
        ysize, xsize = da.shape[-2:]
        if has_species:
            out[:, :, :, iy : iy + ysize, ix : ix + xsize] = da.values
        else:
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
    pattern_species = re.compile(
        r"[\w-]+_c(?P<species>[0-9]+)_(?P<time>[0-9-]+)_l(?P<layer>[0-9]+)_p(?P<subdomain>[0-9]{3})",
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
    species = []
    has_species = False
    for i, p in enumerate(paths):
        if not i:
            if pattern_species.search(paths[0]) is not None:
                has_species = True
                pattern = pattern_species
        search = pattern.search(p)
        timestr = search["time"]
        layer = int(search["layer"])
        number = int(search["subdomain"])
        grouped_by_time[timestr][number].append(p)
        count_per_subdomain[number] += 1
        numbers.append(number)
        layers.append(layer)
        timestrings.append(timestr)
        if has_species:
            species.append(int(search["species"]))

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

    if has_species:
        pattern = r"{name}_c{species}_{time}_l{layer}_p\d+"
    else:
        pattern = r"{name}_{time}_l{layer}_p\d+"
    timestrings = list(grouped_by_time.keys())

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
        # unique also sorts
        coords["time"] = xr.CFTimeIndex(np.unique(times))
    else:
        coords["time"] = np.unique(times)
    if has_species:
        coords["species"] = np.array(sorted(set(species)))
        shape = (
            coords["species"].size,
            1,
            coords["layer"].size,
            coords["y"].size,
            coords["x"].size,
        )
        dims = ("species", "time", "layer", "y", "x")
        time_axis = 1
    else:
        shape = (1, coords["layer"].size, coords["y"].size, coords["x"].size)
        dims = ("time", "layer", "y", "x")
        time_axis = 0

    # Collect and merge data
    merged = []
    dtype = headers[0]["dtype"]
    sorted_order = np.argsort(times)  # get time ordering right before merging
    sorted_timestrings = np.array(timestrings)[sorted_order]
    for timestr in sorted_timestrings:
        group = grouped_by_time[timestr]
        # Build a single array per timestep
        timestep_data = dask.delayed(_merge_subdomains)(group, use_cftime, pattern)
        dask_array = dask.array.from_delayed(timestep_data, shape, dtype=dtype)
        merged.append(dask_array)
    data = dask.array.concatenate(merged, axis=time_axis)

    # Get tops and bottoms if possible
    headers = [header(path, pattern) for path in grouped_by_time[first_time][first]]
    tops = [c.get("top", None) for c in headers]
    bots = [c.get("bot", None) for c in headers]
    layers = [c.get("layer", None) for c in headers]
    _, unique_indices = np.unique(layers, return_index=True)
    all_have_z = all(map(lambda v: v is not None, itertools.chain(tops, bots)))
    if all_have_z:
        if coords["layer"].size > 1:
            coords = array_io.reading._array_z_coord(coords, tops, bots, unique_indices)
        else:
            coords = array_io.reading._scalar_z_coord(coords, tops, bots)

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
    dictionary
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
    d = {}
    for n in unique_names:
        d[n] = []  # prepare empty lists to append to
    for p, n in zip(paths, names):
        d[n].append(p)

    # load each group into a DataArray
    das = [
        array_io.reading._load(v, use_cftime, pattern, _read, header)
        for v in d.values()
    ]

    # store each DataArray under it's own name in a dictionary
    dd = {da.name: da for da in das}
    # Initially I wanted to return a xarray Dataset here,
    # but then realised that it is not always aligned, and therefore not possible, see
    # https://github.com/pydata/xarray/issues/1471#issuecomment-313719395
    # It is not aligned when some parameters only have a non empty subset of a dimension,
    # such as L2 + L3. This dict provides a similar interface anyway. If a Dataset is constructed
    # from unaligned DataArrays it will make copies of the data, which we don't want.
    return dd


def write(path, a, nodata=1.0e20, dtype=np.float32):
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
        raise ValueError(
            f"Dimensions must be exactly ('y', 'x'). Received {a.dims} instead."
        )

    flip = slice(None, None, -1)
    if not a.indexes["x"].is_monotonic_increasing:
        a = a.isel(x=flip)
    if not a.indexes["y"].is_monotonic_decreasing:
        a = a.isel(y=flip)
    # TODO: check is_monotonic, but also for single col/row idfs...

    # Header is fully doubled in size in case of double precision ...
    # This means integers are also turned into 8 bytes
    # and requires padding with some additional bytes
    data_dtype = a.dtype
    if dtype == np.float64:
        if data_dtype != np.float64:
            a = a.astype(np.float64)
        reclenid = 2295
        floatformat = "d"
        intformat = "q"
        doubleprecision = True
    elif dtype == np.float32:
        reclenid = 1271
        floatformat = "f"
        intformat = "i"
        doubleprecision = False
        if data_dtype != np.float32:
            a = a.astype(np.float32)
    else:
        raise ValueError("Invalid dtype, IDF allows only np.float32 and np.float64")

    # Only fillna if data can contain na values
    if (data_dtype == np.float32) or (data_dtype == np.float64):
        a = a.fillna(nodata)

    with f_open(path, "wb") as f:
        f.write(struct.pack("i", reclenid))  # Lahey RecordLength Ident.
        if doubleprecision:
            f.write(struct.pack("i", reclenid))
        nrow = a.y.size
        ncol = a.x.size
        f.write(struct.pack(intformat, ncol))
        f.write(struct.pack(intformat, nrow))

        dx, xmin, xmax, dy, ymin, ymax = util.spatial_reference(a)

        f.write(struct.pack(floatformat, xmin))
        f.write(struct.pack(floatformat, xmax))
        f.write(struct.pack(floatformat, ymin))
        f.write(struct.pack(floatformat, ymax))
        f.write(struct.pack(floatformat, float(a.min())))  # dmin
        f.write(struct.pack(floatformat, float(a.max())))  # dmax
        f.write(struct.pack(floatformat, nodata))

        if isinstance(dx, float) and isinstance(dy, float):
            ieq = True  # equidistant
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
        if doubleprecision:
            f.write(struct.pack("xxxx"))  # not used

        if ieq:
            f.write(struct.pack(floatformat, abs(dx)))
            f.write(struct.pack(floatformat, abs(dy)))
        if itb:
            f.write(struct.pack(floatformat, top))
            f.write(struct.pack(floatformat, bot))
        if not ieq:
            np.abs(a.coords["dx"].values).astype(a.dtype).tofile(f)
            np.abs(a.coords["dy"].values).astype(a.dtype).tofile(f)
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


def save(path, a, nodata=1.0e20, pattern=None, dtype=np.float32):
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
    dtype : type, ``{np.float32, np.float64}``, default is ``np.float32``.
        Whether to write single precision (``np.float32``) or double precision
        (``np.float64``) IDF files.

    Example
    -------
    Consider a DataArray ``da`` that has dimensions ``('layer', 'y', 'x')``, with the
    layer dimension consisting of layer 1 and 2:

    >>> imod.idf.save('path/to/head', da)

    This writes the following two IDF files: ``path/to/head_l1.idf`` and
    ``path/to/head_l2.idf``.

    To disable adding coordinates to the files, specify ``pattern="{name}"``:

    >>> imod.idf.save('path/to/head', da, pattern="{name}")

    The ".idf" extension will always be added automatically.

    It is possible to generate custom filenames using a format string. The
    default filenames would be generated by the following format string:

    >>> imod.idf.save("example", pattern="{name}_l{layer}{extension}")

    If you desire zero-padded numbers that show up neatly sorted in a
    file manager, you may specify:

    >>> imod.idf.save("example", pattern="{name}_l{layer:02d}{extension}")

    In this case, a 0 will be padded for single digit numbers ('1' will become
    '01').

    To get a date with dashes, use the following pattern:

    >>> pattern="{name}_{time:%Y-%m-%d}_l{layer}{extension}"

    """

    # Cast datatype if necessary
    if dtype not in (np.float32, np.float64):
        raise ValueError("Invalid dtype, IDF allows only np.float32 and np.float64")

    # Swap coordinates if possible, add "dz" if possible.
    a = _as_voxeldata(a)

    # Deal with path
    path = pathlib.Path(path)

    if path.suffix == "":
        path = path.with_suffix(".idf")

    array_io.writing._save(path, a, nodata, pattern, dtype, write)
