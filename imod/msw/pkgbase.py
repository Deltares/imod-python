import abc
import textwrap
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from imod.fixed_format import format_fixed_width


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

    def write(self, directory: Union[str, Path], index: np.ndarray, svat: xr.DataArray):
        directory = Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f, index, svat)

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

    def write_dataframe_fixed_width(self, file, dataframe):
        for row in dataframe.itertuples():
            for index, metadata in enumerate(self._metadata_dict.values()):
                content = format_fixed_width(row[index + 1], metadata)
                file.write(content)
            file.write("\n")

    def _index_da(self, da, index):
        return da.values.ravel()[index]

    def _render(self, file, index, svat):
        data_dict = {"svat": svat.values.ravel()[index]}

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

        return self.write_dataframe_fixed_width(file, dataframe)

    def _pkgcheck(self):
        for var in self._with_subunit:
            if "subunit" not in self.dataset[var].coords:
                raise ValueError(
                    textwrap.dedent(
                        f"""Variable '{var}' in {self.__class__} should contain
                         'subunit' coordinate"""
                    )
                )
        for var in self._without_subunit:
            if "subunit" in self.dataset[var].coords:
                raise ValueError(
                    textwrap.dedent(
                        f"""Variable '{var}' in {self.__class__} should not
                         contain 'subunit' coordinate"""
                    )
                )
