import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cftime
import numpy as np
from imod import util


def test_compose():
    d = {
        "name": "head",
        "directory": Path("path", "to"),
        "extension": ".idf",
        "layer": 5,
        "time": cftime.DatetimeProlepticGregorian(2018, 2, 22, 9, 6, 57),
    }
    path = util.compose(d)
    assert isinstance(path, Path)
    targetpath = Path(d["directory"], "head_20180222090657_l5.idf")
    assert path == targetpath


def test_decompose():
    d = util.decompose("path/to/head_20180222090657_l5.idf")
    refd = OrderedDict(
        [
            ("extension", ".idf"),
            ("directory", Path("path", "to")),
            ("name", "head"),
            ("time", cftime.DatetimeProlepticGregorian(2018, 2, 22, 9, 6, 57)),
            ("layer", 5),
        ]
    )
    assert isinstance(d, OrderedDict)
    assert d == refd


def test_decompose_dateonly():
    d = util.decompose("20180222090657.idf")
    refd = OrderedDict(
        [
            ("extension", ".idf"),
            ("directory", Path(".")),
            ("name", "20180222090657"),
            ("time", cftime.DatetimeProlepticGregorian(2018, 2, 22, 9, 6, 57)),
        ]
    )
    assert isinstance(d, OrderedDict)
    assert d == refd


def test_compose_year9999():
    d = {
        "name": "head",
        "directory": Path("path", "to"),
        "extension": ".idf",
        "layer": 5,
        "time": cftime.DatetimeProlepticGregorian(9999, 2, 22, 9, 6, 57),
    }
    path = util.compose(d)
    assert isinstance(path, Path)
    targetpath = Path(d["directory"], "head_99990222090657_l5.idf")
    assert path == targetpath


def test_decompose_dateonly_year9999():
    d = util.decompose("99990222090657.idf")
    refd = OrderedDict(
        [
            ("extension", ".idf"),
            ("directory", Path(".")),
            ("name", "99990222090657"),
            ("time", cftime.DatetimeProlepticGregorian(9999, 2, 22, 9, 6, 57)),
        ]
    )
    assert isinstance(d, OrderedDict)
    assert d == refd
