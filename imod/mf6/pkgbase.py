import abc
import numbers
import pathlib
from typing import Any, Mapping, Optional

import numpy as np
import xarray as xr
from xarray.core.utils import is_scalar
import xugrid as xu

import imod
from imod.mf6.interfaces.ipackagebase import IPackageBase
from imod.typing.grid import (
    GridDataArray,
    GridDataset,
    is_spatial_grid,
    merge_with_dictionary,
)

TRANSPORT_PACKAGES = ("adv", "dsp", "ssm", "mst", "ist", "src")
EXCHANGE_PACKAGES = ("gwfgwf", "gwfgwt", "gwtgwt")

def _is_scalar_nan(da: GridDataArray):
    """
    Test if is_scalar_nan, carefully avoid loading grids in memory
    """
    if is_scalar(da):
        stripped_value = da.values[()]
        return isinstance(stripped_value, numbers.Real) and np.isnan(stripped_value)  # type: ignore[call-overload]
    return False



class PackageBase(IPackageBase, abc.ABC):
    """
    This class is used for storing a collection of Xarray DataArrays or UgridDataArrays
    in a dataset. A load-from-file method is also provided. Storing to file is done by calling
    object.dataset.to_netcdf(...)
    """

    # This method has been added to allow mock.patch to mock created objects
    # https://stackoverflow.com/questions/64737213/how-to-patch-the-new-method-of-a-class
    def __new__(cls, *_, **__):
        return super(PackageBase, cls).__new__(cls)

    def __init__(
        self, variables_to_merge: Mapping[str, GridDataArray | float | int | bool | str]
    ):
        # Merge variables, perform exact join to verify if coordinates values
        # are consistent amongst variables.
        self.__dataset = merge_with_dictionary(variables_to_merge, join="exact")

    @property
    def dataset(self) -> GridDataset:
        return self.__dataset

    @dataset.setter
    def dataset(self, value: GridDataset) -> None:
        self.__dataset = value

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def __setitem__(self, key, value):
        self.dataset.__setitem__(key, value)

    def to_netcdf(
        self, *args, mdal_compliant: bool = False, crs: Optional[Any] = None, **kwargs
    ):
        """

        Write dataset contents to a netCDF file. Custom encoding rules can be
        provided on package level by overriding the _netcdf_encoding in the
        package

        Parameters
        ----------
        *args:
            Will be passed on to ``xr.Dataset.to_netcdf`` or
            ``xu.UgridDataset.to_netcdf``.
        mdal_compliant: bool, optional
            Convert data with
            :func:`imod.prepare.spatial.mdal_compliant_ugrid2d` to MDAL
            compliant unstructured grids. Defaults to False.
        crs: Any, optional
            Anything accepted by rasterio.crs.CRS.from_user_input
            Requires ``rioxarray`` installed.
        **kwargs:
            Will be passed on to ``xr.Dataset.to_netcdf`` or
            ``xu.UgridDataset.to_netcdf``.

        """
        kwargs.update({"encoding": self._netcdf_encoding()})

        dataset = self.dataset
        if isinstance(dataset, xu.UgridDataset):
            if mdal_compliant:
                dataset = dataset.ugrid.to_dataset()
                mdal_dataset = imod.util.spatial.mdal_compliant_ugrid2d(
                    dataset, crs=crs
                )
                mdal_dataset.to_netcdf(*args, **kwargs)
            else:
                dataset.ugrid.to_netcdf(*args, **kwargs)
        else:
            if is_spatial_grid(dataset):
                dataset = imod.util.spatial.gdal_compliant_grid(dataset, crs=crs)
            dataset.to_netcdf(*args, **kwargs)

    def _netcdf_encoding(self):
        """

        The encoding used in the to_netcdf method
        Override this to provide custom encoding rules

        """
        return {}

    @classmethod
    def _from_dataset(cls, ds: GridDataset):
        """
        Create package from dataset. Note that no initialization validation is
        done.
        """
        instance = cls.__new__(cls)
        instance.dataset = ds
        return instance

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
            dataset = imod.util.spatial.from_mdal_compliant_ugrid2d(dataset)
            # Drop node dim, as we don't need in the UgridDataset (it will be
            # preserved in the ``grid`` property, so topology stays intact!)
            node_dim = dataset.grid.node_dimension
            dataset = dataset.drop_dims(node_dim)

        # Replace NaNs by None
        for key, value in dataset.items():
            if _is_scalar_nan(value):
                dataset[key] = None

        return cls._from_dataset(dataset)
