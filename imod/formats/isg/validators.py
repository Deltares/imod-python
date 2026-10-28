from enum import IntEnum
from typing import Any

import numpy as np
from pydantic import AfterValidator


def member_of(enum: type[IntEnum]) -> AfterValidator:
    """Check that all values of an integer vector are members of ``enum``."""
    allowed = np.array([member.value for member in enum])

    def check(array: np.ndarray) -> np.ndarray:
        invalid = np.setdiff1d(array, allowed)
        if invalid.size:
            raise ValueError(f"invalid {enum.__name__} values: {invalid.tolist()}")
        return array

    return AfterValidator(check)


def check_records(name: str, first: np.ndarray, count: np.ndarray, n: int) -> None:
    """Check that ``count`` records from ``first`` lie within records 1 to ``n``."""
    invalid = (count < 0) | ((count > 0) & ((first < 1) | (first + count - 1 > n)))
    if invalid.any():
        rows = (np.flatnonzero(invalid) + 1).tolist()
        raise ValueError(f"{name}: records {rows} reference outside 1 to {n}")


def to_vector(value: Any, dtype: Any) -> np.ndarray:
    """A read-only, one-dimensional numpy array of ``dtype`` (any if None)."""
    if isinstance(value, np.ndarray) and dtype is not None:
        array = value.astype(dtype, casting="same_kind", copy=False)
    else:
        array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"expected a 1D array, got {array.ndim}D")
    array = array.view()
    array.flags.writeable = False
    return array
