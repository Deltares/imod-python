import numbers

from numpy.typing import DTypeLike


def is_float(dtype: DTypeLike) -> bool:
    return issubclass(dtype.type, numbers.Real)


def is_integer(dtype: DTypeLike) -> bool:
    return issubclass(dtype.type, numbers.Integral)
