import abc
import inspect
import numbers
import pathlib
from functools import wraps
from typing import Callable

import numpy as np
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.interfaces.ipackagebase import IPackageBase
from imod.typing.grid import GridDataset, merge_with_dictionary

TRANSPORT_PACKAGES = ("adv", "dsp", "ssm", "mst", "ist", "src")
EXCHANGE_PACKAGES = "gwfgwf"
ARGS_TO_EXCLUDE = ["validate"]


def pkg_init(exclude_in_dataset: list[str]):
    # Nested decorator to support arguments, see:
    # https://stackoverflow.com/a/42581103
    def pkg_init_decorator(init: Callable):
        @wraps(init)
        def merge_init(self, *args, **kwargs):
            # Collect user specified args and kwargs
            args_dict = (
                inspect.signature(self.__init__).bind_partial(*args, **kwargs).arguments
            )
            # Add default values
            for k, v in inspect.signature(self.__init__).parameters.items():
                if k not in args_dict and v.default is not inspect.Parameter.empty:
                    args_dict[k] = v.default
            # Create dict of args to be added to dataset by removing args in
            # exclude_in_dataset.
            args_dict_dataset = {
                key: value
                for key, value in args_dict.items()
                if key not in exclude_in_dataset
            }

            super(type(self), self).__init__(args_dict_dataset)
            return init(self, **args_dict)

        return merge_init

    return pkg_init_decorator


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

    def __init__(self, variables_to_merge=None):
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

    def to_netcdf(self, *args, **kwargs):
        """

        Write dataset contents to a netCDF file.
        Custom encoding rules can be provided on package level by overriding the _netcdf_encoding in the package

        """
        kwargs.update({"encoding": self._netcdf_encoding()})
        self.dataset.to_netcdf(*args, **kwargs)

    def _netcdf_encoding(self):
        """

        The encoding used in the to_netcdf method
        Override this to provide custom encoding rules

        """
        return {}

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
