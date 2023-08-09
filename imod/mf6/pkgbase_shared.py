import abc
from typing import Any

import numpy as np
import xarray as xr
import xugrid as xu


class NewPackageBase(abc.ABC):
    """
    TODO: Consider making this an abstract base class for both high-level as well as low-level packages.
    """

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

    def _valid(self, value):
        """
        Filters values that are None, False, or a numpy.bool_ False.
        Needs to be this specific, since 0.0 and 0 are valid values, but are
        equal to a boolean False.
        """
        # Test singletons
        if value is False or value is None:
            return False
        # Test numpy bool (not singleton)
        elif isinstance(value, np.bool_) and not value:
            return False
        # When dumping to netCDF and reading back, None will have been
        # converted into a NaN. Only check NaN if it's a floating type to avoid
        # TypeErrors.
        elif np.issubdtype(type(value), np.floating) and np.isnan(value):
            return False
        else:
            return True

    def copy(self) -> Any:
        # All state should be contained in the dataset.
        return type(self)(**self.dataset.copy())
