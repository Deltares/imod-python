from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from imod.common.utilities.dtype import is_bool, is_float, is_integer

FLOAT_DTYPES = [
    np.dtype("float16"),
    np.dtype("float32"),
    np.dtype("float64"),
    pd.Float64Dtype(),
]
INTEGER_DTYPES = [
    np.dtype("int8"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
    pd.Int8Dtype(),
    pd.Int16Dtype(),
    pd.Int32Dtype(),
    pd.Int64Dtype(),
]
BOOL_DTYPES = [np.dtype("bool"), pd.BooleanDtype()]
UNSUPPORTED_DTYPES = [
    np.dtype("str"),
    pd.StringDtype(),
    np.dtype("datetime64"),
    pd.DatetimeTZDtype("s", ZoneInfo("UTC")),
    pd.IntervalDtype(),
    pd.PeriodDtype("20d"),
]


def test_is_float():
    for dtype in FLOAT_DTYPES:
        assert is_float(dtype)

    for dtype in INTEGER_DTYPES + BOOL_DTYPES + UNSUPPORTED_DTYPES:
        assert not is_float(dtype)


def test_is_integer():
    for dtype in INTEGER_DTYPES:
        assert is_integer(dtype)

    for dtype in FLOAT_DTYPES + BOOL_DTYPES + UNSUPPORTED_DTYPES:
        assert not is_integer(dtype)


def test_is_bool():
    for dtype in BOOL_DTYPES:
        assert is_bool(dtype)

    for dtype in FLOAT_DTYPES + INTEGER_DTYPES + UNSUPPORTED_DTYPES:
        assert not is_bool(dtype)
