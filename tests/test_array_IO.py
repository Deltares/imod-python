import cftime
import numpy as np
import pytest

from imod.array_IO import reading


def test_to_nan():
    a = np.array([1.0, 2.000001, np.nan, 4.0])
    c = reading._to_nan(a, np.nan)
    assert np.allclose(c, a, equal_nan=True)
    c = reading._to_nan(a, 2.0)
    b = np.array([1.0, np.nan, np.nan, 4.0])
    assert np.allclose(c, b, equal_nan=True)


def test_check_cellsizes():
    # (h["dx"], h["dy"])
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.000001, 3.0])
    c = np.array([1.0, 2.1, 3.0])
    d = np.array([2.0, 2.000001, 2.0])
    e = np.array([4.0, 5.0, 6.0])
    f = np.array([4.0, 5.0, 6.0, 7.0])
    # length one always checks out
    reading._check_cellsizes([(2.0, 3.0)])
    # floats only
    reading._check_cellsizes([(2.0, 3.0), (2.0, 3.0)])
    reading._check_cellsizes([(2.0, 3.0), (2.000001, 3.0)])
    # ndarrays only
    reading._check_cellsizes([(a, e), (a, e)])
    # different length a and f
    reading._check_cellsizes([(a, f), (a, f)])
    reading._check_cellsizes([(a, e), (b, e)])
    # mix of floats and ndarrays
    reading._check_cellsizes([(2.0, d)])
    with pytest.raises(ValueError, match="Cellsizes of IDFs do not match"):
        # floats only
        reading._check_cellsizes([(2.0, 3.0), (2.1, 3.0)])
        # ndarrays only
        reading._check_cellsizes([(a, e), (c, e)])
        # mix of floats and ndarrays
        reading._check_cellsizes([(2.1, d)])
        # Unequal lengths
        reading._check_cellsizes([(a, e), (f, e)])


def test_has_dim():
    t = cftime.DatetimeProlepticGregorian(2019, 2, 28)
    assert reading._has_dim([t, 2, 3])
    assert not reading._has_dim([None, None, None])
    with pytest.raises(ValueError):
        reading._has_dim([t, 2, None])