"""
Module for utilities related to data types of different kinds of origins.
np.issubdtype unfortunately is too restrictive for some of our use cases, as it
does not for example consider pandas extension dtypes.
"""

import numbers
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import DTypeLike

from imod.typing import GridDataArray


class SupportsDTypeType(Protocol):
    @property
    def type(self) -> type[Any]: ...


def is_float(dtype: DTypeLike) -> bool:
    try:
        return np.issubdtype(dtype, np.floating)
    except TypeError:
        # Catch cases where dtype is not a numpy dtype and check if subclass is
        # not an integer. As numpy-style integers are also considered real
        # numbers.
        dtype_with_type = cast(SupportsDTypeType, dtype)
        return issubclass(dtype_with_type.type, numbers.Real) and not issubclass(
            dtype_with_type.type, numbers.Integral
        )


def is_integer(dtype: DTypeLike) -> bool:
    try:
        return np.issubdtype(dtype, np.integer)
    except TypeError:
        dtype_with_type = cast(SupportsDTypeType, dtype)
        return issubclass(dtype_with_type.type, numbers.Integral)


def is_bool(dtype: DTypeLike) -> bool:
    try:
        return np.issubdtype(dtype, np.bool_)
    except TypeError:
        dtype_with_type = cast(SupportsDTypeType, dtype)
        return issubclass(dtype_with_type.type, np.bool)


def as_string(da: GridDataArray) -> GridDataArray:
    """
    Convert a GridDataArray to a string type if it is of a string-like dtype.
    This function checks if the dtype of the input array is string-like (i.e.,
    kind "S", "U", "O" or contains "StringDType"). If so, it attempts to convert
    the array to a plain string dtype. If the conversion fails due to the
    presence of a StringDType, it first converts the array to an object dtype
    and then to a string dtype.

    This is a TEMPORARY workaround, and will NOT be necessary in numpy version
    2.6.0, when this fix is released: https://github.com/numpy/numpy/pull/32097

    Parameters
    ----------
    da : GridDataArray
        The input array to convert.

    Returns
    -------
    GridDataArray
        The converted array with string dtype.
    """
    if da.dtype.kind in ("S", "U", "O") or "StringDType" in str(da.dtype):
        try:
            da = da.astype(str)
        except TypeError as e:
            # numpy/xarray with StringDType cannot cast directly to plain `str`
            # (unspecified fixed-width unicode). Convert via object first.
            if "StringDType" not in str(e):
                raise e
            da = da.astype(object).astype(str)
    return da
