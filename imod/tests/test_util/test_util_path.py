import datetime
import pathlib
import re

import cftime
import numpy as np

import imod


def test_compose():
    d = {
        "name": "head",
        "directory": pathlib.Path("path", "to"),
        "extension": ".idf",
        "layer": 5,
        "time": datetime.datetime(2018, 2, 22, 9, 6, 57),
        "species": 6,
    }
    path = imod.util.path.compose(d)
    targetpath = pathlib.Path(d["directory"], "head_c6_20180222090657_l5.idf")
    assert path == targetpath

    d.pop("species")
    path = imod.util.path.compose(d)
    targetpath = pathlib.Path(d["directory"], "head_20180222090657_l5.idf")
    assert path == targetpath

    d.pop("layer")
    path = imod.util.path.compose(d)
    targetpath = pathlib.Path(d["directory"], "head_20180222090657.idf")
    assert path == targetpath

    d.pop("time")
    d["layer"] = 1
    path = imod.util.path.compose(d)
    targetpath = pathlib.Path(d["directory"], "head_l1.idf")
    assert path == targetpath

    d["species"] = 6
    path = imod.util.path.compose(d)
    targetpath = pathlib.Path(d["directory"], "head_c6_l1.idf")
    assert path == targetpath


def test_compose__pattern():
    d = {
        "name": "head",
        "directory": pathlib.Path("path", "to"),
        "extension": ".foo",
        "layer": 5,
    }
    targetpath = pathlib.Path(d["directory"], "head_2018-02-22_l05.foo")

    d["time"] = datetime.datetime(2018, 2, 22, 9, 6, 57)
    path = imod.util.path.compose(
        d, pattern="{name}_{time:%Y-%m-%d}_l{layer:02d}{extension}"
    )
    assert path == targetpath

    d["time"] = cftime.DatetimeProlepticGregorian(2018, 2, 22, 9, 6, 57)
    path = imod.util.path.compose(
        d, pattern="{name}_{time:%Y-%m-%d}_l{layer:02d}{extension}"
    )
    assert path == targetpath

    d["time"] = np.datetime64("2018-02-22 09:06:57")
    path = imod.util.path.compose(
        d, pattern="{name}_{time:%Y-%m-%d}_l{layer:02d}{extension}"
    )
    assert path == targetpath

    targetpath = pathlib.Path(d["directory"], ".foo_makes_head_no_layer5_sense_day22")
    path = imod.util.path.compose(
        d, pattern="{extension}_makes_{name}_no_layer{layer:d}_sense_day{time:%d}"
    )
    assert path == targetpath


def test_decompose():
    d = imod.util.path.decompose("path/to/head_20180222090657_l5.idf")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("path", "to"),
        "name": "head",
        "time": datetime.datetime(2018, 2, 22, 9, 6, 57),
        "layer": 5,
        "dims": ["time", "layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_species():
    d = imod.util.path.decompose("path/to/conc_c3_20180222090657_l5.idf")
    refd = {
        "extension": ".idf",
        "species": 3,
        "directory": pathlib.Path("path", "to"),
        "name": "conc",
        "time": datetime.datetime(2018, 2, 22, 9, 6, 57),
        "layer": 5,
        "dims": ["species", "time", "layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_short_date():
    d = imod.util.path.decompose("path/to/head_20180222_l5.idf")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("path", "to"),
        "name": "head",
        "time": datetime.datetime(2018, 2, 22),
        "layer": 5,
        "dims": ["time", "layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_nonstandard_date():
    d = imod.util.path.decompose("path/to/head_2018-02-22_l5.idf")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("path", "to"),
        "name": "head",
        "time": datetime.datetime(2018, 2, 22),
        "layer": 5,
        "dims": ["time", "layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_only_year():
    d = imod.util.path.decompose(
        "path/to/head_2018_l5.idf", pattern="{name}_{time}_l{layer}"
    )
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("path", "to"),
        "name": "head",
        "time": datetime.datetime(2018, 1, 1),
        "layer": 5,
        "dims": ["time", "layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_underscore():
    d = imod.util.path.decompose("path/to/starting_head_20180222090657_l5.idf")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("path", "to"),
        "name": "starting_head",
        "time": datetime.datetime(2018, 2, 22, 9, 6, 57),
        "layer": 5,
        "dims": ["time", "layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_dash():
    d = imod.util.path.decompose("path/to/starting-head_20180222090657_l5.idf")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("path", "to"),
        "name": "starting-head",
        "time": datetime.datetime(2018, 2, 22, 9, 6, 57),
        "layer": 5,
        "dims": ["time", "layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_steady_state():
    d = imod.util.path.decompose("path/to/head_steady-state_l64.idf")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("path", "to"),
        "name": "head",
        "time": "steady-state",
        "layer": 64,
        "dims": ["layer", "time"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_underscore_in_name():
    d = imod.util.path.decompose("path/to/some_name.idf")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("path", "to"),
        "name": "some_name",
        "dims": [],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_pattern_underscore():
    d = imod.util.path.decompose(
        "path/to/starting_head_20180222090657_l5.idf", pattern="{name}_{time}_l{layer}"
    )
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("path", "to"),
        "name": "starting_head",
        "time": datetime.datetime(2018, 2, 22, 9, 6, 57),
        "layer": 5,
        "dims": ["time", "layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_pattern_dash():
    d = imod.util.path.decompose(
        "path/to/starting-head_20180222090657_l5.idf", pattern="{name}_{time}_l{layer}"
    )
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("path", "to"),
        "name": "starting-head",
        "time": datetime.datetime(2018, 2, 22, 9, 6, 57),
        "layer": 5,
        "dims": ["time", "layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_regexpattern():
    pattern = re.compile(r"(?P<name>[\w]+)L(?P<layer>[\d+]*)", re.IGNORECASE)
    d = imod.util.path.decompose("headL11.idf", pattern=pattern)
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("."),
        "name": "head",
        "layer": 11,
        "dims": ["layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_nodate():
    d = imod.util.path.decompose("dem_10m.idf")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("."),
        "name": "dem_10m",
        "dims": [],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_dateonly():
    d = imod.util.path.decompose("20180222090657.idf", pattern="{time}")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("."),
        "name": "20180222090657",
        "time": datetime.datetime(2018, 2, 22, 9, 6, 57),
        "dims": ["time"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_datelayeronly():
    d = imod.util.path.decompose("20180222090657_l7.idf", pattern="{time}_l{layer}")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("."),
        "name": "20180222090657_7",
        "time": datetime.datetime(2018, 2, 22, 9, 6, 57),
        "layer": 7,
        "dims": ["time", "layer"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_decompose_z_float():
    d = imod.util.path.decompose("test_0.25.idf", pattern="{name}_{z}")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("."),
        "name": "test",
        "z": "0.25",
        "dims": ["z"],
    }
    assert isinstance(d, dict)
    assert d == refd


def test_compose_year9999():
    d = {
        "name": "head",
        "directory": pathlib.Path("path", "to"),
        "extension": ".idf",
        "layer": 5,
        "time": datetime.datetime(9999, 2, 22, 9, 6, 57),
        "dims": ["time"],
    }
    path = imod.util.path.compose(d)
    targetpath = pathlib.Path(d["directory"], "head_99990222090657_l5.idf")
    assert path == targetpath


def test_decompose_dateonly_year9999():
    d = imod.util.path.decompose("99990222090657.idf", pattern="{time}")
    refd = {
        "extension": ".idf",
        "directory": pathlib.Path("."),
        "name": "99990222090657",
        "time": datetime.datetime(9999, 2, 22, 9, 6, 57),
        "dims": ["time"],
    }
    assert isinstance(d, dict)
    assert d == refd
