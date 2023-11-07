"""
Module to define type aliases.
"""

from typing import TypeAlias, Union

import xarray as xr
import xugrid as xu

GridDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
GridDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
ScalarDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
