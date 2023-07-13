import abc
import numbers
import pathlib

import numpy as np
import xarray as xr
import xugrid as xu

import imod

TRANSPORT_PACKAGES = ("adv", "dsp", "ssm", "mst", "ist", "src")


def dis_recarr(arrdict, layer, notnull):
    # Define the numpy structured array dtype
    index_spec = [("layer", np.int32), ("row", np.int32), ("column", np.int32)]
    field_spec = [(key, np.float64) for key in arrdict]
    sparse_dtype = np.dtype(index_spec + field_spec)
    # Initialize the structured array
    nrow = notnull.sum()
    recarr = np.empty(nrow, dtype=sparse_dtype)
    # Fill in the indices
    if notnull.ndim == 2:
        recarr["row"], recarr["column"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, irow, icolumn = np.argwhere(notnull).transpose()
        recarr["row"] = irow + 1
        recarr["column"] = icolumn + 1
        recarr["layer"] = layer[ilayer]
    return recarr


def disv_recarr(arrdict, layer, notnull):
    # Define the numpy structured array dtype
    index_spec = [("layer", np.int32), ("cell2d", np.int32)]
    field_spec = [(key, np.float64) for key in arrdict]
    sparse_dtype = np.dtype(index_spec + field_spec)
    # Initialize the structured array
    nrow = notnull.sum()
    recarr = np.empty(nrow, dtype=sparse_dtype)
    # Fill in the indices
    if notnull.ndim == 1 and layer.size == 1:
        recarr["cell2d"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, icell2d = np.argwhere(notnull).transpose()
        recarr["cell2d"] = icell2d + 1
        recarr["layer"] = layer[ilayer]
    return recarr


class PackageBase(abc.ABC):
    def __init__(self, allargs=None):
        if allargs is not None:
            for arg in allargs.values():
                if isinstance(arg, xu.UgridDataArray):
                    self.dataset = xu.UgridDataset(grids=arg.ugrid.grid)
                    return
        self.dataset = xr.Dataset()

    """
    This class is used for storing a collection of Xarray dataArrays or ugrid-DataArrays
    in a dataset. A load-from-file method is also provided. Storing to file is done by calling
    object.dataset.to_netcdf(...)
    """

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def __setitem__(self, key, value):
        self.dataset.__setitem__(key, value)

    @classmethod
    def from_file(cls, path, **kwargs):
        """
        Loads an imod mf6 package from a file (currently only netcdf and zarr are supported).
        Note that it is expected that this file was saved with imod.mf6.Package.dataset.to_netcdf(),
        as the checks upon package initialization are not done again!

        Parameters
        ----------
        path : str, pathlib.Path
            Path to the file.
        **kwargs : keyword arguments
            Arbitrary keyword arguments forwarded to ``xarray.open_dataset()``, or
            ``xarray.open_zarr()``.
        Refer to the examples.

        Returns
        -------
        package : imod.mf6.Package
            Returns a package with data loaded from file.

        Examples
        --------

        To load a package from a file, e.g. a River package:

        >>> river = imod.mf6.River.from_file("river.nc")

        For large datasets, you likely want to process it in chunks. You can
        forward keyword arguments to ``xarray.open_dataset()`` or
        ``xarray.open_zarr()``:

        >>> river = imod.mf6.River.from_file("river.nc", chunks={"time": 1})

        Refer to the xarray documentation for the possible keyword arguments.
        """
        path = pathlib.Path(path)
        if path.suffix in (".zip", ".zarr"):
            # TODO: seems like a bug? Remove str() call if fixed in xarray/zarr
            dataset = xr.open_zarr(str(path), **kwargs)
        else:
            dataset = xr.open_dataset(path, **kwargs)

        if dataset.ugrid_roles.topology:
            dataset = xu.UgridDataset(dataset)
            dataset = imod.util.from_mdal_compliant_ugrid2d(dataset)

        # Replace NaNs by None
        for key, value in dataset.items():
            stripped_value = value.values[()]
            if isinstance(stripped_value, numbers.Real) and np.isnan(stripped_value):
                dataset[key] = None

        instance = cls.__new__(cls)
        instance.dataset = dataset
        return instance
