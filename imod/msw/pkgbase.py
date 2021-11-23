import abc

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

    @staticmethod
    def _check_range(dataframe, metadata_dict):
        for varname in dataframe:
            minimum = metadata_dict[varname].minimum
            maximum = metadata_dict[varname].maximum
            if (dataframe[varname] < minimum).any() or (
                dataframe[varname] > maximum
            ).any():
                raise ValueError(
                    f"{varname}: not all values are within range ({minimum}-{maximum})."
                )

    @staticmethod
    def number_format_fixed_width(width: int, dtype, metadata):
        if dtype == "string":
            return "{:" + f"{width}" + "}"
        elif np.issubdtype(dtype, np.integer):
            return "{:" + f"{width}d" + "}"
        elif np.issubdtype(dtype, np.floating):
            decimal_number_width = max(0, width - metadata.whole_number_digits - 2)
            return "{:" + f"+{width}.{decimal_number_width}f" + "}"
        else:
            raise TypeError(f"dtype {dtype} is not supported")

    @classmethod
    def write_dataframe_fixed_width(cls, file, dataframe, metadata_dict) -> str:
        formatter = {}
        for index, (varname, metadata) in enumerate(metadata_dict.items()):
            if index == 0:
                width = metadata.column_width
            else:
                # Compensate space that will be added by `to_string`
                # for all but the first column
                width = metadata.column_width - 1

            number_format = cls.number_format_fixed_width(
                width, dataframe[varname].dtypes, metadata
            )

            formatter[varname] = number_format.format

        return dataframe.to_string(
            file, index=False, header=False, formatters=formatter, justify="right"
        )


class MetaData:
    def __init__(self, column_width, minimum, maximum):
        self.column_width = column_width
        self.minimum = minimum
        self.maximum = maximum

        if self.minimum is not None and self.maximum is not None:
            self.calc_whole_number_digits()

    def calc_whole_number_digits(self):
        max_abs = max(abs(self.minimum), abs(self.maximum))
        self.whole_number_digits = len(str(int(max_abs)))
