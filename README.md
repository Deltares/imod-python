Work with [iMOD](http://oss.deltares.nl/web/imod) MODFLOW models.

## Getting started
In absence of actual documentation, here are some of the functions:
```python
import imod.io

arr, meta = imod.io.readidf('ibound.idf')
imod.io.writeidf('ibound-out.idf', arr, meta)

df = imod.io.readipf('wells.ipf')
imod.io.writeipf(df, 'wells-out.ipf')

# get all calculated heads in a xarray DataArray
# with dimensions time, layer, row, column and coordinates y and x
da = imod.io.loaddata('path/to/results', 'head_*.idf')

# The aim is to couple iMOD data files and xarray more tightly
# such that we can lazily import an entire model into an xarray
# Dataset, do calculations/aggregations/modifications on the
# Dataset, and write everything back to the iMOD data files.
```
