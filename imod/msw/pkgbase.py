import abc
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, TextIO, TypeAlias, Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from imod.common.utilities.clip import clip_spatial_box, clip_time_slice
from imod.common.utilities.dataclass_type import DataclassType, EmptyRegridMethod
from imod.common.utilities.regrid import (
    _regrid_like,
)
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.mf6_wel_adapter import Mf6Wel
from imod.msw.fixed_format import format_fixed_width
from imod.typing import IntArray
from imod.typing.grid import GridDataArray, GridDataset
from imod.util.regrid import RegridderWeightsCache

DataDictType: TypeAlias = dict[str, IntArray | int | str]


class MetaSwapPackage(abc.ABC):
    """
    MetaSwapPackage is used to share methods for Metaswap packages.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.
    """

    __slots__ = "_pkg_id"
    _file_name = "filename_not_set"
    _regrid_method: DataclassType = EmptyRegridMethod()
    _with_subunit: tuple = ()
    _without_subunit: tuple = ()
    _to_fill: tuple = ()
    _metadata_dict: dict = {}

    def __init__(self):
        self.dataset = xr.Dataset()

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def __setitem__(self, key, value):
        self.dataset.__setitem__(key, value)

    @property
    def dataset(self) -> GridDataset:
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    @classmethod
    def _from_dataset(cls, ds: GridDataset):
        """
        Create package from dataset. Note that no initialization is done.
        """
        instance = cls.__new__(cls)
        instance.dataset = ds
        return instance

    def write(
        self,
        directory: Union[str, Path],
        index: IntArray,
        svat: xr.DataArray,
        mf6_dis: StructuredDiscretization,
        mf6_well: Mf6Wel,
    ):
        """
        Write MetaSWAP package to its corresponding fixed format file. This has
        the `.inp` extension.

        Parameters
        ----------
        directory: string and Path
            Directory to write package in.
        index: numpy array of integers
            Array of integers with indices.
        svat: xr.DataArray
            Grid with svats, has dimensions "subunit, y, x".
        mf6_dis: StructuredDiscretization (optional)
            Modflow 6 structured discretization.
        mf6_well: Mf6Wel (optional)
            If given, this parameter describes sprinkling of SVAT units from MODFLOW
            cells.
        """
        directory = Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f, index, svat, mf6_dis, mf6_well)

    def _check_range(self, dataframe):
        """
        Check if provided data does not exceeds MetaSWAPs ranges. These ranges
        are specified in the ``self._metadata_dict`` for each variable.
        """
        for varname in dataframe:
            min_value = self._metadata_dict[varname].min_value
            max_value = self._metadata_dict[varname].max_value
            if (dataframe[varname] < min_value).any() or (
                dataframe[varname] > max_value
            ).any():
                raise ValueError(
                    f"{varname}: not all values are within range ({min_value}-{max_value})."
                )

    def _write_dataframe_fixed_width(self, file, dataframe):
        """Write dataframe to fixed format file."""
        for row in dataframe.itertuples():
            for index, metadata in enumerate(self._metadata_dict.values()):
                content = format_fixed_width(row[index + 1], metadata)
                file.write(content)
            file.write("\n")

    def _index_da(self, da: xr.DataArray, index: IntArray) -> np.ndarray:
        """
        Helper method that converts a DataArray to a 1d numpy array, and
        consequently applies boolean indexing.
        """
        return da.values.ravel()[index]

    def _render(
        self,
        file: TextIO,
        index: IntArray,
        svat: xr.DataArray,
        mf6_dis: StructuredDiscretization,
        mf6_well: Mf6Wel,
    ) -> None:
        """
        Collect to be written data in a DataFrame and call
        ``self.write_dataframe_fixed_width``
        """
        data_dict: DataDictType = {"svat": svat.values.ravel()[index]}

        subunit = svat.coords["subunit"]

        for var in self._with_subunit:
            data_dict[var] = self._index_da(self.dataset[var], index)

        for var in self._without_subunit:
            da = self.dataset[var].expand_dims(subunit=subunit)
            data_dict[var] = self._index_da(da, index)

        for var in self._to_fill:
            data_dict[var] = ""

        dataframe = pd.DataFrame(
            data=data_dict, columns=list(self._metadata_dict.keys())
        )

        self._check_range(dataframe)

        return self._write_dataframe_fixed_width(file, dataframe)

    def _pkgcheck(self):
        """
        Method to do package checks. The base class version checks if provided
        data has a subunit coordinate or not.
        """
        for var in self._with_subunit:
            if "subunit" not in self.dataset[var].coords:
                raise ValueError(
                    f"Variable '{var}' in {self.__class__} should contain "
                    "'subunit' coordinate"
                )
        for var in self._without_subunit:
            if "subunit" in self.dataset[var].coords:
                raise ValueError(
                    f"Variable '{var}' in {self.__class__} should not "
                    "contain 'subunit' coordinate"
                )

    def _valid(self, value):
        return True

    def _get_non_grid_data(self, grid_names: list[str]) -> dict[str, Any]:
        """
        This function copies the attributes of a dataset that are scalars, such as options.

        parameters
        ----------
        grid_names: list of str
            the names of the attribbutes of a dataset that are grids.
        """
        result = {}
        all_non_grid_data = list(self.dataset.keys())
        for name in (
            gridname for gridname in grid_names if gridname in all_non_grid_data
        ):
            all_non_grid_data.remove(name)
        for name in all_non_grid_data:
            result[name] = self.dataset[name]
        return result

    @property
    def auxiliary_data_fields(self) -> dict[str, str]:
        return {}

    def _is_regridding_supported(self) -> bool:
        return True

    def _is_grid_agnostic_package(self) -> bool:
        """
        Check if the package is grid agnostic, meaning it does not depend on a
        specific grid structure.
        """
        return False

    def regrid_like(
        self,
        target_grid: GridDataArray,
        regrid_cache: RegridderWeightsCache,
        regridder_types: Optional[DataclassType] = None,
    ) -> "MetaSwapPackage":
        """
        Creates a package of the same type as this package, based on another
        discretization. It regrids all the arrays in this package to the desired
        discretization, and leaves the options unmodified. At the moment only
        regridding to a different planar grid is supported, meaning
        ``target_grid`` has different ``"x"`` and ``"y"`` or different
        ``cell2d`` coords.

        The default regridding methods are specified in the ``_regrid_method``
        attribute of the package. These defaults can be overridden using the
        input parameters of this function.

        Parameters
        ----------
        target_grid: xr.DataArray or xu.UgridDataArray
            a grid defined using the same discretization as the one we want to
            regrid the package to.
        regrid_cache: RegridderWeightsCache
            stores regridder weights for different regridders. Can be used to
            speed up regridding, if the same regridders are used several times
            for regridding different arrays.
        regridder_types: RegridMethodType, optional
            dictionary mapping arraynames (str) to a tuple of regrid type (a
            specialization class of BaseRegridder) and function name (str) this
            dictionary can be used to override the default mapping method.

        Examples
        --------
        To regrid the infiltration package with a non-default method for the
        infiltration capacity, call ``regrid_like`` with these arguments:

        >>> regridder_types = imod.msw.regrid.InfiltrationRegridMethod(infiltration_capacity=(imod.RegridderType.OVERLAP, "max"))
        >>> regrid_cache = imod.util.regrid.RegridderWeightsCache()
        >>> new_infiltration = infiltration.regrid_like(like, regrid_cache, regridder_types)

        Returns
        -------
        A package with the same options as this package, and with all the
        data-arrays regridded to another discretization, similar to the one used
        in input argument "target_grid"
        """
        try:
            result = _regrid_like(self, target_grid, regrid_cache, regridder_types)
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"package could not be regridded:{e}")
        return result

    def clip_box(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> "MetaSwapPackage":
        """
        Clip a package by a bounding box (time, layer, y, x).

        Slicing intervals may be half-bounded, by providing None:

        * To select 500.0 <= x <= 1000.0:
          ``clip_box(x_min=500.0, x_max=1000.0)``.
        * To select x <= 1000.0: ``clip_box(x_min=None, x_max=1000.0)``
          or ``clip_box(x_max=1000.0)``.
        * To select x >= 500.0: ``clip_box(x_min = 500.0, x_max=None.0)``
          or ``clip_box(x_min=1000.0)``.

        Parameters
        ----------
        time_min: optional
        time_max: optional
        x_min: optional, float
        x_max: optional, float
        y_min: optional, float
        y_max: optional, float

        Returns
        -------
        clipped: Package
        """
        if not self._is_clipping_supported():
            raise ValueError("this package does not support clipping.")

        selection = self.dataset
        selection = clip_time_slice(selection, time_min=time_min, time_max=time_max)
        selection = clip_spatial_box(
            selection,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        cls = type(self)
        return cls._from_dataset(selection)

    @classmethod
    def get_regrid_methods(cls) -> DataclassType:
        """
        Returns the default regrid methods for this package. You can use modify
        to customize the regridding of the package.

        Returns
        -------
        DataclassType
            The regrid methods for this package, which is a dataclass with
            attributes that are tuples of (regridder type, method name). If no
            regrid methods are defined, returns an instance of
            EmptyRegridMethod.

        Examples
        --------
        Get the regrid methods for the Drainage package:

        >>> regrid_settings = Infiltration.get_regrid_methods()

        You can modify the regrid methods by changing the attributes of the
        returned dataclass instance. For example, to set the regridding method
        for ``infiltration_capacity`` to minimum.

        >>> regrid_settings.infiltration_capacity = (imod.RegridderType.OVERLAP, "min")

        These settings can then be used to regrid the package:

        >>> infiltration.regrid_like(like, regridder_types=regrid_settings)

        """
        return deepcopy(cls._regrid_method)

    def from_imod5_data(self, *args, **kwargs):
        raise NotImplementedError("Method not implemented for this package.")

    def _is_clipping_supported(self) -> bool:
        return True
