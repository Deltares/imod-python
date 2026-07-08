"""
Module for utilities related to data types of different kinds of origins.
np.issubdtype unfortunately is too restrictive for some of our use cases, as it
does not for example consider pandas extension dtypes.
"""

import numbers

import numpy as np


def is_float(dtype: np.dtype) -> bool:
    try:
        return np.issubdtype(dtype, np.floating)
    except TypeError:
        # Catch cases where dtype is not a numpy dtype and check if subclass is
        # not an integer. As numpy-style integers are also considered real
        # numbers.
        return issubclass(dtype.type, numbers.Real) and not issubclass(
            dtype.type, numbers.Integral
        )


def is_integer(dtype: np.dtype) -> bool:
    try:
        return np.issubdtype(dtype, np.integer)
    except TypeError:
        return issubclass(dtype.type, numbers.Integral)


def is_bool(dtype: np.dtype) -> bool:
    try:
        return np.issubdtype(dtype, np.bool_)
    except TypeError:
        return issubclass(dtype.type, np.bool)
