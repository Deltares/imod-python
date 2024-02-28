"""
Module to define type aliases.
"""

from typing import TypeAlias, Union

import xarray as xr
import xugrid as xu

GridDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
GridDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
ScalarDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
ScalarDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
UnstructuredData: TypeAlias = Union[xu.UgridDataset, xu.UgridDataArray]
