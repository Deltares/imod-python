"""
Module to define type aliases.
"""

from typing import TypeAlias, Union

import numpy as np
import xarray as xr
import xugrid as xu

GridDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
GridDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
ScalarAsDataArray: TypeAlias = Union[xr.DataArray, xu.UgridDataArray]
ScalarAsDataset: TypeAlias = Union[xr.Dataset, xu.UgridDataset]
UnstructuredData: TypeAlias = Union[xu.UgridDataset, xu.UgridDataArray]
FloatArray: TypeAlias = np.ndarray
IntArray: TypeAlias = np.ndarray
