import collections
import glob
import itertools
import pathlib
from collections.abc import Callable, Sequence
from re import Pattern
from typing import Any, TypeAlias, cast

import dask
import numpy as np
import xarray as xr
from numpy.typing import DTypeLike

from imod.util import nested_dict, spatial, time

PatternType: TypeAlias = Pattern[str] | str
PathType: TypeAlias = str | pathlib.Path
BoundsType: TypeAlias = list[tuple[float, float, float, float]]
CellsizesType: TypeAlias = list[tuple[float | np.ndarray, float | np.ndarray]]
ZInterfaceType: TypeAlias = list[float | None]


def _all_equal(seq: list[Any], elem: str) -> None:
    """Raise an error if not all elements of a list are equal"""
    if not seq.count(seq[0]) == len(seq):
        raise ValueError(f"All {elem} must be the same, found: {set(seq)}")


def _check_cellsizes(cellsizes: CellsizesType) -> None:
    """
    Checks if cellsizes match, raises ValueError otherwise

    Parameters
    ----------
    cellsizes : list of tuples
        tuples may contain:
        * two floats, dx and dy, for equidistant files
        * two ndarrays, dx and dy, for nonequidistant files

    Returns
    -------
    None
    """
    msg = "Cellsizes of IDFs do not match"
    if len(cellsizes) == 1:
        return None
    try:
        if not (cellsizes.count(cellsizes[0]) == len(cellsizes)):
            raise ValueError(msg)
    except ValueError:  # contains ndarrays
        try:
            # all ndarrays
            dx0, dy0 = cellsizes[0]
            for dx, dy in cellsizes[1:]:
                if np.allclose(dx0, dx) and np.allclose(dy0, dy):
                    pass
                else:
                    raise ValueError(msg)
        except ValueError:
            # some ndarrays, some floats
            # create floats for comparison with allclose
            dx = np.atleast_1d(cellsizes[0][0])[0]
            dy = np.atleast_1d(cellsizes[0][1])[0]
            # comparison
            for cellsize in cellsizes:
                # Unfortunately this allocates by broadcasting dx and dy
                if not np.allclose(cellsize[0], dx):
                    raise ValueError(msg)
                if not np.allclose(cellsize[1], dy):
                    raise ValueError(msg)


def _has_dim(seq: list[None | float]) -> bool:
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


def _to_nan(a: np.ndarray, nodata: float) -> np.ndarray:
    """Change all nodata values in the array to NaN"""
    # it needs to be NaN for xarray to deal with it properly
    # no need to store the nodata value if it is always NaN
    if np.isnan(nodata):
        return a
    else:
        isnodata = np.isclose(a, nodata)
        a[isnodata] = np.nan
        return a


def _array_z_coord(
    coords: dict[str, Any],
    tops: list[float],
    bots: list[float],
    unique_indices: np.ndarray,
) -> dict[str, Any]:
    top = np.array(tops)[unique_indices]
    bot = np.array(bots)[unique_indices]
    dz = top - bot
    z = top - 0.5 * dz
    if top[0] > top[1]:  # decreasing coordinate
        dz *= -1.0
    if np.allclose(dz, dz[0]):
        coords["dz"] = dz[0]
    else:
        coords["dz"] = ("layer", dz)
    coords["z"] = ("layer", z)
    return coords


def _scalar_z_coord(coords: dict[str, Any], tops: list[float], bots: list[float]):
    # They must be all the same to be used, as they cannot be assigned
    # to layer.
    top: np.ndarray = np.unique(tops)
    bot: np.ndarray = np.unique(bots)
    if top.size == bot.size == 1:
        top = top.item()
        bot = bot.item()
        dz = top - bot
        z = top - 0.5 * dz
        coords["dz"] = float(dz)  # cast from array
        coords["z"] = float(z)
    return coords


def _initialize_groupby(ndims: int):
    """
    This function generates a data structure consisting of defaultdicts, to use
    for grouping arrays by dimension. The number of dimensions may vary, so the
    degree of nesting might vary as well.

    For a single dimension such as layer, it'll look like:
    d = {1: da1, 2: da2, etc.}

    For two dimensions, layer and time:
    d = {"2001-01-01": {1: da1, 2: da3}, "2001-01-02": {1: da3, 2: da4}, etc.}

    And so on for more dimensions.

    Defaultdicts are very well suited to this application. The
    itertools.groupby object does not provide any benefits in this case, it
    simply provides a generator; its entries have to come presorted. It also
    does not provide tools for these kind of variably nested groupby's.

    Pandas.groupby does provide this functionality. However, pandas dataframes
    do not accept any field value, whereas these dictionaries do. Might be
    worthwhile to look into, if performance is an issue.

    Parameters
    ----------
    ndims : int
        Number of dimensions

    Returns
    -------
        groupby : Defaultdicts, ndims - 1 times nested
    """
    return nested_dict.initialize_nested_dict(ndims - 1)


def _ndconcat(arrays: list[list[Any]], ndim: int) -> dask.array.Array:
    """
    Parameters
    ----------
    arrays : list of lists, n levels deep.
        E.g.  [[da1, da2], [da3, da4]] for n = 2.
        (compare with docstring for _initialize_groupby)
    ndim : int
        number of dimensions over which to concatenate.

    Returns
    -------
    concatenated : xr.DataArray
        Input concatenated over n dimensions.
    """
    if ndim == 1:  # base case
        return dask.array.stack(arrays, axis=0)
    else:  # recursive case
        ndim = ndim - 1
        out = [_ndconcat(arrays_in, ndim) for arrays_in in arrays]
        return dask.array.stack(out, axis=0)


def _dims_coordinates(
    header_coords: dict[str, list[Any]],
    bounds: BoundsType,
    cellsizes: CellsizesType,
    tops: ZInterfaceType,
    bots: ZInterfaceType,
    use_cftime: bool,
) -> tuple[list[str], dict[str, Any]]:
    # create coordinates
    coords = spatial._xycoords(bounds[0], cellsizes[0])
    dims = ["y", "x"]
    # Time and layer have to be special cased.
    # Time due to the multitude of datetimes possible
    # Layer because layer is required to properly assign top and bot data.
    haslayer = False
    hastime = False
    otherdims = []
    for dim in list(header_coords.keys()):
        if dim == "layer":
            haslayer = True
            coords["layer"], unique_indices = np.unique(
                header_coords["layer"], return_index=True
            )
        elif dim == "time":
            hastime = True
            times, use_cftime = time._convert_datetimes(
                header_coords["time"], use_cftime
            )
            if use_cftime:
                coords["time"] = xr.CFTimeIndex(np.unique(times))
            else:
                coords["time"] = np.unique(times)
        else:
            otherdims.append(dim)
            coords[dim] = np.unique(header_coords[dim])

    # Ensure right dimension
    if haslayer:
        dims.insert(0, "layer")
    if hastime:
        dims.insert(0, "time")
    for dim in otherdims:
        dims.insert(0, dim)

    # Deal with voxel idf top and bottom data
    all_have_z = all((v is not None for v in itertools.chain(tops, bots)))
    if all_have_z:
        tops_float = cast(list[float], tops)
        bots_float = cast(list[float], bots)
        if haslayer and coords["layer"].size > 1:
            coords = _array_z_coord(coords, tops_float, bots_float, unique_indices)
        else:
            coords = _scalar_z_coord(coords, tops_float, bots_float)

    return dims, coords


def _dask(
    path: PathType,
    attrs: dict[str, Any] | None = None,
    pattern: PatternType | None = None,
    _read: Callable[[str, int, int, int, Any, DTypeLike], np.ndarray] | None = None,
    header: Callable[[PathType, PatternType | None], dict[str, Any]] | None = None,
) -> tuple[dask.array.Array, dict[str, Any]]:
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
        to the examples in ``imod.idf.open``.
    _read : Callable[[str, int, int, int, Any, DTypeLike], np.ndarray], optional
        Function to read individual files. Must be provided.
    header : Callable[[PathType, PatternType | None], dict[str, Any]], optional
        Function to read the header of a file. Must be provided if attrs is not
        supplied.

    Returns
    -------
    dask.array
        A float32 dask.array with shape (nrow, ncol) of the values
        in the IDF file. On opening all nodata values are changed
        to NaN in the dask.array.
    dict
        A dict with all metadata.
    """

    path = pathlib.Path(path)

    if attrs is None:
        if header is None:
            raise ValueError(
                "Header function must be provided if attrs is not supplied."
            )
        attrs = header(path, pattern)
    # If we don't unpack, it seems we run into trouble with the dask array later
    # on, probably because attrs isn't immutable. This works fine instead.
    headersize = attrs.pop("headersize")
    nrow = attrs["nrow"]
    ncol = attrs["ncol"]
    dtype = attrs["dtype"]
    # In case of floating point data, nodata is always represented by nan.
    if "float" in dtype:
        nodata = attrs.pop("nodata")
    else:
        nodata = attrs["nodata"]

    # Dask delayed caches the input arguments. If the working directory changes
    # before .compute(), the file cannot be found if the path is relative.
    abspath = path.resolve()
    # dask.delayed requires currying
    a = dask.delayed(_read)(abspath, headersize, nrow, ncol, nodata, dtype)
    x = dask.array.from_delayed(a, shape=(nrow, ncol), dtype=dtype)
    return x, attrs


def load_as_components(
    paths: Sequence[PathType],
    use_cftime: bool,
    _read: Callable[..., np.ndarray],
    headers: list[dict[str, Any]],
) -> tuple[dask.array.Array, dict[str, Any], list[str], str]:
    """
    Combine a list of paths to IDFs to components that can be used to construct
    a single xarray.DataArray.

    Parameters
    ----------
    paths : list[str]
        List of file paths to IDF files.
    use_cftime : bool
        Whether to use cftime for time coordinates.
    _read : Callable
        Function to read individual IDF files.
    headers : list[dict]
        List of headers corresponding to each IDF file.

    Returns
    -------
    dask.array
        Dask array containing the combined data from all IDF files.
    dict
        Dictionary of coordinates for the DataArray.
    list[str]
        List of dimension names for the DataArray.
    str
        Name of the DataArray.
    """
    # this function also works for single IDFs
    names: list[str] = [h["name"] for h in headers]
    _all_equal(names, "names")

    # Extract data from headers
    bounds: BoundsType = []
    cellsizes: CellsizesType = []
    tops: ZInterfaceType = []
    bots: ZInterfaceType = []
    header_coords: dict[str, list[Any]] = collections.defaultdict(list)
    for h in headers:
        bounds.append((h["xmin"], h["xmax"], h["ymin"], h["ymax"]))
        cellsizes.append((h["dx"], h["dy"]))
        tops.append(h.get("top", None))
        bots.append(h.get("bot", None))
        for key in h["dims"]:
            header_coords[key].append(h[key])
    # Do a basic check whether IDFs align in x and y
    _all_equal(bounds, "bounding boxes")
    _check_cellsizes(cellsizes)
    # Generate coordinates
    dims, coords = _dims_coordinates(
        header_coords, bounds, cellsizes, tops, bots, use_cftime
    )
    # This part have makes use of recursion to deal with an arbitrary number
    # of dimensions. It may therefore be a little hard to read.
    groupbydims = dims[:-2]  # skip y and x
    ndim = len(groupbydims)
    groupby = _initialize_groupby(ndim)
    if ndim == 0:  # Single idf
        dask_array, _ = _dask(paths[0], headers[0], _read=_read)
    else:
        for path, attrs in zip(paths, headers):
            da, _ = _dask(path, attrs=attrs, _read=_read)
            groupbykeys = [attrs[k] for k in groupbydims]
            nested_dict.set_nested(groupby, groupbykeys, da)
        dask_arrays = nested_dict.sorted_nested_dict(groupby)
        dask_array = _ndconcat(dask_arrays, ndim)
    return dask_array, coords, dims, names[0]


def _load(
    paths: Sequence[PathType],
    use_cftime: bool,
    _read: Callable[..., np.ndarray],
    headers: list[dict[str, Any]],
) -> xr.DataArray:
    dask_array, coords, dims, name = load_as_components(
        paths, use_cftime, _read, headers
    )
    out = xr.DataArray(dask_array, coords, dims, name=name)

    first_attrs = headers[0]

    if "crs" in first_attrs:
        out.attrs["crs"] = first_attrs["crs"]
    if "nodata" in first_attrs:
        out.attrs["nodata"] = first_attrs["nodata"]

    return out


def handle_path(
    path: pathlib.Path | str | list[pathlib.Path | str],
) -> list[pathlib.Path]:
    """
    Handle a path input and return a list of pathlib.Path objects.

    Parameters
    ----------
    path : pathlib.Path, str, or list of pathlib.Path
        The input path(s) to handle.

    Returns
    -------
    list[pathlib.Path]
        List of pathlib.Path objects corresponding to the input path(s).

    Raises
    ------
    FileNotFoundError
        If no files matching the input path(s) are found.
    """
    if isinstance(path, pathlib.Path):
        path = str(path)

    if isinstance(path, list):
        paths = [pathlib.Path(p) for p in path]
    else:
        # Sort: glob order is filesystem dependent (alphabetical on NTFS, not
        # on e.g. APFS or ext4), and downstream merging can depend on order.
        paths = [pathlib.Path(p) for p in sorted(glob.glob(path))]

    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"Could not find any files matching {path}")

    return paths


def _open(path, use_cftime, pattern, header, _read):
    paths = handle_path(path)

    headers = [header(p, pattern) for p in paths]

    return _load(paths, use_cftime, _read, headers)
