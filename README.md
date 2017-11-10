Work with [iMOD](http://oss.deltares.nl/web/imod) MODFLOW models in Python.

## Getting started
In absence of actual documentation, here are some of the functions:
```python
import imod

df = imod.io.readipf('wells.ipf')
imod.io.writeipf(df, 'wells-out.ipf')

# get all calculated heads in a xarray DataArray
# with dimensions time, layer, y, x
da = imod.io.loadarray('path/to/results/head_*.idf')

# The aim is to couple iMOD data files and xarray more tightly
# such that we can lazily import an entire model into an xarray
# Dataset, do calculations/aggregations/modifications on the
# Dataset, and write everything back to the iMOD data files.
```

## Notes

- The iMOD 4.1 release will crash on loading NaN nodata values. In earlier and later releases this is not an issue.

## Implementation of `loadarray`

1. Do a `glob` search on all files matching the input path, e.g. `head_*.idf`
2. For each IDF:
  1. Gather all metadata from filename and IDF header
  2. Load IDF array with `np.memmap` in `r+` mode.
  3. Set nodata to `np.nan` (changes input file, done to be compatible with xarray)
  4. Load `np.memmap` into a `dask.array`
  5. Combine metadata and `dask.array` into an `xarray.DataArray`
3. Combine the indivual `DataArray`s into one, adding `time` and `layer` dimensions as necessary.

With the current design, I'm not sure what the added benefit of using dask arrays is, and whether it
should be removed. One possible benefit is that it is only possible to write to the `DataArray`
after calling `.load()` on it first, which in this case still doesn't load it into memory.

We can also still explore using the `c` *copy on write* mode of `numpy.memmap`.

Comment by xarray author Stephan Hoyer about using `np.memmap` on [GitHub](https://github.com/dask/dask/issues/1562#issuecomment-248681863):

> If it's already on disk in a memory-mappable format, it's very hard to
imagine beating np.memmap.
