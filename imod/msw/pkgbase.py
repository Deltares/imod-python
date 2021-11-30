import abc
import os
from dataclasses import dataclass
from numbers import Number

import numpy as np
import xarray as xr


class Package(abc.ABC):
    """
    Package is used to share methods for specific packages with no time
    component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.
    """

    __slots__ = "_pkg_id"

    def __init__(self):
        self.dataset = xr.Dataset()

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def __setitem__(self, key, value):
        self.dataset.__setitem__(key, value)

    def isel(self):
        raise NotImplementedError(
            f"Selection on packages not yet supported. "
            f"To make a selection on the xr.Dataset, call {self._pkg_id}.dataset.isel instead. "
            f"You can create a new package with a selection by calling {__class__.__name__}(**{self._pkg_id}.dataset.isel(**selection))"
        )

    def sel(self):
        raise NotImplementedError(
            f"Selection on packages not yet supported. "
            f"To make a selection on the xr.Dataset, call {self._pkg_id}.dataset.sel instead. "
            f"You can create a new package with a selection by calling {__class__.__name__}(**{self._pkg_id}.dataset.sel(**selection))"
        )

    @abc.abstractmethod
    def write(self, directory):
        return

    def _check_range(self, dataframe):
        for varname in dataframe:
            min_value = self._metadata_dict[varname].min_value
            max_value = self._metadata_dict[varname].max_value
            if (dataframe[varname] < min_value).any() or (
                dataframe[varname] > max_value
            ).any():
                raise ValueError(
                    f"{varname}: not all values are within range ({min_value}-{max_value})."
                )

    @staticmethod
    def format_fixed_width(value, metadata):
        if metadata.dtype == str:
            format_string = "{:" + f"{metadata.column_width}" + "}"
        elif metadata.dtype == int:
            format_string = "{:" + f"{metadata.column_width}d" + "}"
        elif metadata.dtype == float:
            whole_number_digits = len(str(int(abs(value))))
            decimal_number_width = max(
                0, metadata.column_width - whole_number_digits - 2
            )
            format_string = (
                "{:" + f"+{metadata.column_width}.{decimal_number_width}f" + "}"
            )
        else:
            raise TypeError(f"dtype {metadata.dtype} is not supported")

        converted_value = metadata.dtype(value)
        return format_string.format(converted_value)

    def write_dataframe_fixed_width(self, file, dataframe):
        for row in dataframe.itertuples():
            for index, (varname, metadata) in enumerate(self._metadata_dict.items()):
                content = self.format_fixed_width(row[index + 1], metadata)
                file.write(content)
            file.write(os.linesep)

    def _get_preprocessed_array(self, varname, mask, extend_subunits=None):
        array = self.dataset[varname]
        if extend_subunits is not None:
            array = array.expand_dims(subunit=extend_subunits)

        # Apply mask
        if mask is not None:
            array = array.where(mask)

        # Convert to numpy array and flatten it
        array = array.to_numpy().ravel()

        # Remove NaN values
        array = array[~np.isnan(array)]

        return array


@dataclass
class VariableMetaData:
    column_width: int
    min_value: Number
    max_value: Number
    dtype: type
