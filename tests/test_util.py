import os
import pathlib
import collections
from datetime import datetime

import cftime
import numpy as np
import pytest

from imod.io import util


def test_compose():
    d = {
        "name": "head",
        "directory": pathlib.Path("path", "to"),
        "extension": ".idf",
        "layer": 5,
        "time": datetime(2018, 2, 22, 9, 6, 57),
    }
    path = util.compose(d)
    assert isinstance(path, pathlib.Path)
    targetpath = pathlib.Path(d["directory"], "head_20180222090657_l5.idf")
    assert path == targetpath


def test_decompose():
    d = util.decompose("path/to/head_20180222090657_l5.idf")
    refd = collections.OrderedDict(
        [
            ("extension", ".idf"),
            ("directory", pathlib.Path("path", "to")),
            ("name", "head"),
            ("time", datetime(2018, 2, 22, 9, 6, 57)),
            ("layer", 5),
        ]
    )
    assert isinstance(d, collections.OrderedDict)
    assert d == refd


def test_decompose_dateonly():
    d = util.decompose("20180222090657.idf")
    refd = collections.OrderedDict(
        [
            ("extension", ".idf"),
            ("directory", pathlib.Path(".")),
            ("name", "20180222090657"),
            ("time", datetime(2018, 2, 22, 9, 6, 57)),
        ]
    )
    assert isinstance(d, collections.OrderedDict)
    assert d == refd


def test_compose_year9999():
    d = {
        "name": "head",
        "directory": pathlib.Path("path", "to"),
        "extension": ".idf",
        "layer": 5,
        "time": datetime(9999, 2, 22, 9, 6, 57),
    }
    path = util.compose(d)
    assert isinstance(path, pathlib.Path)
    targetpath = pathlib.Path(d["directory"], "head_99990222090657_l5.idf")
    assert path == targetpath


def test_decompose_dateonly_year9999():
    d = util.decompose("99990222090657.idf")
    refd = collections.OrderedDict(
        [
            ("extension", ".idf"),
            ("directory", pathlib.Path(".")),
            ("name", "99990222090657"),
            ("time", datetime(9999, 2, 22, 9, 6, 57)),
        ]
    )
    assert isinstance(d, collections.OrderedDict)
    assert d == refd


def test_datetime_conversion__withinbounds():
    times = [datetime(y, 1, 1) for y in range(2000, 2010)]
    converted, use_cftime = util._convert_datetimes(times, use_cftime=False)
    assert use_cftime == False
    assert all(t.dtype == "<M8[ns]" for t in converted)
    assert converted[0] == np.datetime64("2000-01-01", "ns")
    assert converted[-1] == np.datetime64("2009-01-01", "ns")


def test_datetime_conversion__outofbounds():
    times = [datetime(y, 1, 1) for y in range(1670, 1680)]
    with pytest.warns(UserWarning):
        converted, use_cftime = util._convert_datetimes(times, use_cftime=False)
    assert use_cftime == True
    assert all(type(t) == cftime.DatetimeProlepticGregorian for t in converted)
    assert converted[0] == cftime.DatetimeProlepticGregorian(1670, 1, 1)
    assert converted[-1] == cftime.DatetimeProlepticGregorian(1679, 1, 1)


def test_datetime_conversion__withinbounds_cftime():
    times = [datetime(y, 1, 1) for y in range(2000, 2010)]
    converted, use_cftime = util._convert_datetimes(times, use_cftime=True)
    assert use_cftime == True
    assert all(type(t) == cftime.DatetimeProlepticGregorian for t in converted)
    assert converted[0] == cftime.DatetimeProlepticGregorian(2000, 1, 1)
    assert converted[-1] == cftime.DatetimeProlepticGregorian(2009, 1, 1)
