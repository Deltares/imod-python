import abc
import numbers
from pathlib import Path
from typing import Any, Mapping, Optional, Self, final

import numpy as np
import xarray as xr
import xugrid as xu
from xarray.core.utils import is_scalar

import imod
from imod.common.interfaces.ipackagebase import IPackageBase
from imod.common.serializer import EngineType, create_package_serializer
from imod.typing.grid import (
    GridDataArray,
    GridDataset,
    merge_with_dictionary,
)

TRANSPORT_PACKAGES = ("adv", "dsp", "ssm", "mst", "ist", "src")
EXCHANGE_PACKAGES = ("gwfgwf", "gwfgwt", "gwtgwt")
UTIL_PACKAGES = ("ats", "hpc")


def _is_scalar_nan(da: GridDataArray):
    """
    Test if is_scalar_nan, carefully avoid loading grids in memory
    """
    scalar_data: bool = is_scalar(da)
    if scalar_data:
        stripped_value = da.to_numpy()[()]
        return isinstance(stripped_value, numbers.Real) and np.isnan(stripped_value)  # type: ignore[call-overload]
    return False


class PackageBase(IPackageBase, abc.ABC):
    """
    This class is used for storing a collection of Xarray DataArrays or UgridDataArrays
    in a dataset. A load-from-file method is also provided. Storing to file is done by calling
    object.dataset.to_netcdf(...)
    """

    _pkg_id = ""

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

    @property
    def pkg_id(self) -> str:
        return self._pkg_id

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def __setitem__(self, key, value):
        self.dataset.__setitem__(key, value)

    @classmethod
    def _from_dataset(cls, ds: GridDataset) -> Self:
        """
        Create package from dataset. Note that no initialization validation is
        done.
        """
        instance = cls.__new__(cls)
        instance.dataset = ds
        return instance

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> Self:
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
        path = Path(path)
        if path.suffix in (".zip", ".zarr"):
            import zarr

            if path.suffix == ".zip":
                with zarr.storage.ZipStore(path, mode="r") as store:
                    dataset = xr.open_zarr(store, **kwargs)
            else:
                dataset = xr.open_zarr(str(path), **kwargs)
        else:
            dataset = xr.open_dataset(path, chunks="auto", **kwargs)

        if dataset.ugrid_roles.topology:
            dataset = xu.UgridDataset(dataset)
            dataset = imod.util.spatial.from_mdal_compliant_ugrid2d(dataset)
            # Drop node dim, as we don't need in the UgridDataset (it will be
            # preserved in the ``grid`` property, so topology stays intact!)
            node_dim = dataset.grid.node_dimension
            dataset = dataset.drop_dims(node_dim, errors="ignore")

        # Replace NaNs by None
        for key, value in dataset.items():
            if _is_scalar_nan(value):
                dataset[key] = None

        # to_netcdf converts strings into NetCDF "variable‑length UTF‑8 strings"
        # which are loaded as dtype=object arrays. This will convert them back
        # to str.
        vars = ["species"]
        for var in vars:
            if var in dataset:
                dataset[var] = dataset[var].astype(str)

        return cls._from_dataset(dataset)

    @final
    def to_file(
        self,
        modeldirectory: Path,
        pkgname: str,
        mdal_compliant: bool = False,
        crs: Optional[Any] = None,
        engine: EngineType = "netcdf4",
        **kwargs,
    ) -> Path:
        """
        Write package to file.

        Parameters
        ----------
        modeldirectory : pathlib.Path
            Directory where to write the package file.
        pkgname : str
            Name of the package, used to create the file name.
        mdal_compliant : bool, optional
            Whether to write the package in MDAL-compliant format. Only used if
            ``engine="netcdf4"``. Default is False.
        crs : optional
            Coordinate reference system to use when writing the package. Only
            used if ``engine="netcdf4"`` Default is None.
        engine : str, optional
            File engine used to write packages. Options are ``'netcdf4'``,
            ``'zarr'``, and ``'zarr.zip'``. NetCDF4 is readable by many other
            softwares, for example QGIS. Zarr is optimized for big data, cloud
            storage and parallel access. The ``'zarr.zip'`` option is an
            experimental option which creates a zipped zarr store in a single
            file, which is easier to copy and automatically compresses data as
            well. Default is ``'netcdf4'``.
        **kwargs : keyword arguments
            Additional keyword arguments forwarded to the package serializer's
            `to_file` method.

        Returns
        -------
        pathlib.Path
            Path to the written package file.
        """

        # All serialization logic is in the package serializers do not override
        # this method (hence final decorator).
        filewriter = create_package_serializer(
            engine, mdal_compliant=mdal_compliant, crs=crs
        )
        path = filewriter.to_file(self, modeldirectory, pkgname, **kwargs)
        return path
