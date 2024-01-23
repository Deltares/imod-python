# %%

import itertools
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import dask
import numpy as np
import xarray as xr

# %%


def check_dtypes(das: List[xr.DataArray]) -> None:
    """Check whether the dtypes of all arrays are the same."""
    dtypes = set(da.dtype for da in das)
    if len(dtypes) != 1:
        raise TypeError(f"DataArrays do not match in dtype: {dtypes}")
    return


def check_sizes(sizes: DefaultDict[str, Set[int]], attribute: str) -> None:
    """Utility for checking a dict of dimension names and sizes. Skips x and y."""
    sizes.pop("x", None)
    sizes.pop("y", None)
    conflicting = {k: v for k, v in sizes.items() if len(v) != 1}
    if conflicting:
        message = (
            f"DataArrays do not match in {attribute} along dimension(s):\n"
            + "\n".join([f"   {k}: {v}" for k, v in conflicting.items()])
        )
        raise ValueError(message)
    return


def check_dims(das: List[xr.DataArray]) -> None:
    all_dims = set(da.dims for da in das)
    if len(all_dims) != 1:
        raise ValueError(
            f"All DataArrays should have exactly the same dimensions. Found: {all_dims}"
        )
    last_dims = das[0].dims[-2:]
    if not last_dims == ("y", "x"):
        raise ValueError(f'Last dimensions must be ("y", "x"). Found: {last_dims}')
    check_dim_sizes(das)


def check_dim_sizes(das: List[xr.DataArray]) -> None:
    """Check whether all non-xy dims are equally sized."""
    sizes = defaultdict(set)
    for da in das:
        for key, value in da.sizes.items():
            sizes[key].add(value)
    check_sizes(sizes, "size")
    return


def check_coords(das: List[xr.DataArray]):
    def drop_xy(coords) -> Dict[str, Any]:
        coords = dict(coords)
        coords.pop("y")
        coords.pop("x")
        return xr.Coordinates(coords)

    first_coords = drop_xy(das[0].coords)
    disjoint = [
        i + 1
        for i, da in enumerate(das[1:])
        if not first_coords.equals(drop_xy(da.coords))
    ]
    if disjoint:
        raise ValueError(
            f"Non x-y coordinates do not match for partition 0 with partitions: {disjoint}"
        )
    return


def check_chunk_sizes(das: List[xr.DataArray]) -> None:
    """Check whether all chunks are equal on non-xy dims."""
    chunks = [da.chunks for da in das]
    iterator = (item is None for item in chunks)
    allnone = all(iterator)
    if allnone:
        return
    if any(iterator) != allnone:
        raise ValueError("Some DataArrays are chunked, while others are not.")

    sizes = defaultdict(set)
    for da in das:
        for key, value in zip(da.dims, da.chunks):
            sizes[key].add(value)
    check_sizes(sizes, "chunks")
    return


def merge_arrays(
    arrays: List[np.ndarray],
    ixs: List[np.ndarray],
    iys: List[np.ndarray],
    yx_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Merge the arrays in the last two (y, x) dimensions.

    Parameters
    ----------
    arrays: list of N np.ndarray
    ixs: list of N np.ndarray of int
        The i-th element are the x indices of the i-th array into the merged
        array.
    iys: list of N np.ndarray of int
        The i-th element are the y indices of the i-th array into the merged
        array.
    yx_shape: tuple of int
        The number of rows and columns of the merged array.

    Returns
    -------
    merged: np.ndarray
    """
    first = arrays[0]
    shape = first.shape[:-2] + yx_shape
    out = np.full(shape, np.nan)
    for a, ix, iy in zip(arrays, ixs, iys):
        ysize, xsize = a.shape[-2:]
        # Create view of partition, see:
        # https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
        out_partition_view = out[..., iy : iy + ysize, ix : ix + xsize]
        # Assign active values to view (updates `out` inplace)
        out_partition_view[...] = np.where(~np.isnan(a), a, out_partition_view)
    return out


def _unique_coords(das: List[xr.DataArray], dim: str) -> xr.DataArray:
    """Collect unique coords in list of dataarrays"""
    return np.unique(np.concatenate([da.coords[dim].values for da in das]))


def _merge_nonequidistant_coords(
    das: List[xr.DataArray], coordname: str, indices: List[np.ndarray], nsize: int
):
    out = np.full((nsize,), np.nan)
    for da, index in zip(das, indices):
        coords = da.coords[coordname]
        out[index : index + coords.size] = coords.values
    return out


def _merge_partitions(das: List[xr.DataArray]) -> xr.DataArray:
    # Do some input checking
    check_dtypes(das)
    check_dims(das)
    check_chunk_sizes(das)
    check_coords(das)

    # Create the x and y coordinates of the merged grid.
    x = _unique_coords(das, "x")
    y = _unique_coords(das, "y")
    nrow = y.size
    ncol = x.size
    # Compute the indices for where the different subdomain parts belong
    # in the merged grid.
    ixs = [np.searchsorted(x, da.x.values[0], side="left") for da in das]
    iys = [nrow - np.searchsorted(y, da.y.values[0], side="right") for da in das]
    yx_shape = (nrow, ncol)

    # Collect coordinates
    first = das[0]
    coords = dict(first.coords)
    coords["x"] = x
    coords["y"] = y[::-1]
    if "dx" in first.coords:
        coords["dx"] = ("x", _merge_nonequidistant_coords(das, "dx", ixs, ncol))
    if "dy" in first.coords:
        coords["dy"] = ("y", _merge_nonequidistant_coords(das, "dy", iys, nrow)[::-1])

    arrays = [da.data for da in das]
    if first.chunks is None:
        # If the data is in memory, merge all at once.
        data = merge_arrays(arrays, ixs, iys, yx_shape)
    else:
        # Iterate over the chunks of the dask array. Collect the chunks
        # from every partition and merge them, chunk by chunk.
        # The delayed merged result is stored as a flat list. These can
        # be directly concatenated into a new dask array if chunking occurs
        # on only the first dimension (e.g. time), but not if chunks exist
        # in multiple dimensions (e.g. time and layer).
        #
        # dask.array.block() is capable of concatenating over multiple
        # dimensions if we feed it a nested list of lists of dask arrays.
        # This is more easily represented by a numpy array of objects
        # (dask arrays), since numpy has nice tooling for reshaping.
        #
        # Normally, we'd append to a list, then convert to numpy array and
        # reshape. However, numpy attempts to join a list of dask arrays into
        # a single large numpy array when initialized. This behavior is not
        # triggered when setting individual elements of the array, so we
        # create the numpy array in advance and set its elements.

        block_shape = das[0].data.blocks.shape[:-2]
        merged_blocks = np.empty(np.prod(block_shape), dtype=object)
        dimension_ranges = [range(size) for size in block_shape]
        for i, index in enumerate(itertools.product(*dimension_ranges)):
            # This is a workaround for python 3.10
            # FUTURE: can be rewritten to arr.blocks[*index, ...] in python 3.11
            index_with_ellipsis = tuple(index) + (...,)
            # arr.blocks provides us access to the chunks of the array.
            arrays_to_merge = [arr.blocks[index_with_ellipsis] for arr in arrays]
            delayed_merged = dask.delayed(merge_arrays)(
                arrays_to_merge, ixs, iys, yx_shape
            )
            dask_merged = dask.array.from_delayed(
                delayed_merged,
                shape=arrays_to_merge[0].shape[:-2] + yx_shape,
                dtype=first.dtype,
            )
            merged_blocks[i] = dask_merged

        # After merging, the xy chunks are always (1, 1)
        reshaped = merged_blocks.reshape(block_shape + (1, 1))
        data = dask.array.block(reshaped.tolist())

    return xr.DataArray(
        data=data,
        coords=coords,
        dims=first.dims,
    )


def merge_partitions(
    das: List[xr.DataArray | xr.Dataset],
) -> xr.Dataset:
    first_item = das[0]
    if isinstance(first_item, xr.Dataset):
        unique_keys = set([key for da in das for key in da.keys()])
        merged_ls = []
        for key in unique_keys:
            merged_ls.append(_merge_partitions([da[key] for da in das]).rename(key))
        return xr.merge(merged_ls)
    elif isinstance(first_item, xr.DataArray):
        # Store name to rename after concatenation
        name = das[0].name
        return _merge_partitions(das).to_dataset(name=name)
    else:
        raise TypeError(
            f"Expected type: xr.DataArray or xr.Dataset, got {type(first_item)}"
        )
