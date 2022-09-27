import pathlib

import xarray as xr

from imod import util


def _write_chunks(a, pattern, d, nodata, dtype, write):
    """
    This function writes one chunk of the DataArray 'a' at a time. This is
    necessary to avoid heavily sub-optimal scheduling by xarray/dask when
    writing data to idf's. The problem appears to be caused by the fact that
    .groupby results in repeated computations for every single IDF chunk
    (time and layer). Specifically, merging several subdomains with
    open_subdomains, and then calling save ends up being extremely slow.

    This functions avoids this by calling compute() on the individual chunk,
    before writing (so the chunk therefore has to fit in memory). 'x' and 'y'
    dimensions are not treated as chunks, as all values over x and y have to
    end up in single IDF file.

    The number of chunks is not known beforehand; it may vary per dimension,
    and the number of dimensions may vary as well. The naive solution to this
    is a variable number of for loops, writting explicitly beforehand. Instead,
    this function uses recursion, selecting one chunk per dimension per time.
    The base case is one where only a single chunks remains, and then the write
    occurs (ignoring chunks in x and y).
    """
    dim = a.dims[0]
    dim_is_xy = (dim == "x") or (dim == "y")
    nochunks = a.chunks is None or max(map(len, a.chunks)) == 1
    if nochunks or dim_is_xy:  # Base case
        a = a.compute()
        extradims = list(filter(lambda dim: dim not in ("y", "x"), a.dims))
        if extradims:
            stacked = a.stack(idf=extradims)
            for coordvals, a_yx in list(stacked.groupby("idf")):
                # Groupby sometimes returns an extra 1-sized "idf" dim
                if "idf" in a_yx.dims:
                    a_yx = a_yx.squeeze("idf")
                # set the right layer/timestep/etc in the dict to make the filename
                d.update(dict(zip(extradims, coordvals)))
                fn = util.compose(d, pattern)
                write(fn, a_yx, nodata, dtype)
        else:
            fn = util.compose(d, pattern)
            write(fn, a, nodata, dtype)
    else:  # recursive case
        for dim, chunksizes in zip(a.dims, a.chunks):
            if len(chunksizes) > 1:
                break
        start = 0
        for chunksize in chunksizes:
            end = start + chunksize
            b = a.isel({dim: slice(start, end)})
            # Recurse
            _write_chunks(b, pattern, d, nodata, dtype, write)
            start = end


def _save(path, a, nodata, pattern, dtype, write):
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
        actual filename(s) using conventions, so it only takes the directory and
        name, and extension from this parameter.
    a : xarray.DataArray
        DataArray to be written. It needs to have dimensions ('y', 'x'), and
        optionally ``layer`` and ``time``.
    nodata : float, optional
        Nodata value in the saved IDF files. Xarray uses nan values to represent
        nodata, but these tend to work unreliably in iMOD(FLOW).
        Defaults to a value of 1.0e20.
    pattern : str
        Format string which defines how to create the filenames. See examples.
    write : function
        Which function to use to write 2D arrays.
    """
    if not isinstance(a, xr.DataArray):
        raise TypeError("Data to save must be an xarray.DataArray")
    if a.dims[-2:] != ("y", "x"):
        raise ValueError(
            'Last two dimensions of DataArray to save must be ("y", "x"), '
            f"found: {a.dims}"
        )

    path = pathlib.Path(path)

    # A more flexible schema might be required to support additional variables
    # such as species, for concentration. The straightforward way is by giving
    # a format string, e.g.: {name}_{time}_l{layer}
    # Find the vars in curly braces, and validate with da.coords
    d = {"extension": path.suffix, "name": path.stem, "directory": path.parent}
    d["directory"].mkdir(exist_ok=True, parents=True)

    # Make sure all non-xy dims are ascending
    # otherwise a.stack fails in _write_chunks.
    # y has to be monotonic decreasing
    flip = slice(None, None, -1)
    for dim in a.dims:
        if dim == "y":
            if not a.indexes[dim].is_monotonic_decreasing:
                a = a.isel({dim: flip})
        else:
            if not a.indexes[dim].is_monotonic_increasing:
                a = a.isel({dim: flip})

    # handle the case where they are not a dim but are a coord
    # i.e. you only have one layer but you did a.assign_coords(layer=1)
    # in this case we do want _l1 in the IDF file name
    check_coords = ["layer", "time"]
    for coord in check_coords:
        if (coord in a.coords) and not (coord in a.dims):
            if coord == "time":
                # .item() gives an integer for datetime64[ns], so convert first.
                val = a.coords[coord].values
                if not val == "steady-state":
                    val = a.coords[coord].values.astype("datetime64[us]").item()
            else:
                val = a.coords[coord].item()
            d[coord] = val

    # stack all non idf dims into one new idf dimension,
    # over which we can then iterate to write all individual idfs
    _write_chunks(a, pattern, d, nodata, dtype, write)
