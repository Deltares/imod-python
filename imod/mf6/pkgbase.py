import abc
import inspect
import numbers
import pathlib
import textwrap

import numpy as np
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.interfaces.ipackagebase import IPackageBase
from imod.typing.grid import GridDataset

TRANSPORT_PACKAGES = ("adv", "dsp", "ssm", "mst", "ist", "src")
EXCHANGE_PACKAGES = "gwfgwf"
ARGS_TO_EXCLUDE = ["validate"]


def merge_unstructured_dataset(variables_to_merge):
    """Work around xugrid issue https://github.com/Deltares/xugrid/issues/179"""

    # Separate variables into list of grids and dict of scalar variables
    grids_ls = []
    scalar_dict = {}
    for name, variable in variables_to_merge.items():
        if isinstance(variable, xu.UgridDataArray):
            grids_ls.append(variable.rename(name))
        else:
            scalar_dict[name] = variable

    # Merge grids
    dataset = xu.merge(grids_ls, join="exact")

    # Assign scalar variables manually
    for name, variable in scalar_dict.items():
        dataset[name] = variable

    return dataset


def merge_with_dictionary(variables_to_merge):
    """
    Merge grid and scalar variables provided in dictionary to dataset. Function
    checks if there is no mixing going on between structured and unstructured
    grids. Also allows running function on dictionary with purely scalars, in
    which case it will call to the xarray function.
    """

    error_msg = textwrap.dedent(
        """
        Received both xr.DataArray and xu.UgridDataArray. This means structured
        grids as well as unstructured grids were provided.
        """
    )

    if variables_to_merge is None:
        return xr.Dataset()

    types = [type(arg) for arg in variables_to_merge.values()]
    has_unstructured = xu.UgridDataArray in types
    has_structured = xr.DataArray in types
    if has_structured and has_unstructured:
        raise TypeError(error_msg)
    if has_unstructured:
        return merge_unstructured_dataset(variables_to_merge)

    return xr.merge([variables_to_merge], join="exact")


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
        self.__dataset = merge_with_dictionary(variables_to_merge)

    def _get_variable_names(self, init_method):
        """
        Return variable names based on the arguments provided to the classes'
        __init__ method. Removes argument names that need to be excluded.
        """
        variable_names = list(inspect.signature(init_method).parameters.keys())
        for var_to_exclude in ARGS_TO_EXCLUDE:
            if var_to_exclude in variable_names:
                variable_names.remove(var_to_exclude)
        return variable_names

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
