import os
import pytest
from imod import ipf
import numpy as np
from collections import OrderedDict
import xarray as xr
import pandas as pd
from pathlib import Path
from glob import glob


def remove(globpath):
    paths = glob(globpath)
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


@pytest.fixture(scope="module")
def write_basic_ipf(request):

    # factory function seems easiest way to parameterize tests
    def _write_basic_ipf(path, delim):
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        ipfstring = (
            "2\n"
            "4\n"
            "X\n"
            "Y\n"
            "Z\n"
            '"City of Holland"\n'
            "0{delim}TXT\n"
            "100.0{delim}435.0{delim}-32.3{delim}Amsterdam\n"
            '553.0{delim}143.0{delim}-7.3{delim}"Den Bosch"\n'
        )
        ipfstring = ipfstring.format(delim=delim)
        with open(path, "w") as f:
            f.write(ipfstring)

    # Use global list to add to a list of paths
    # that were generated during testing?
    def teardown():
        remove("*.ipf")

    request.addfinalizer(teardown)
    return _write_basic_ipf


@pytest.fixture(scope="module")
def write_assoc_ipf(request):
    def _write_assoc_ipf(path, delim, assoc_delim):
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        ipfstring = (
            "2\n"
            "3\n"
            "X\n"
            "Y\n"
            "ID\n"
            "3{delim}txt\n"
            "100.0{delim}435.0{delim}A1000\n"
            "553.0{delim}143.0{delim}B2000\n"
        )
        assoc_string = (
            "2\n"
            "2{delim}1\n"
            "time{delim}-999.0\n"
            "level{delim}-999.0\n"
            "20180101000000{delim}1.0\n"
            "20180102000000{delim}-999.0\n"
        )
        ipfstring = ipfstring.format(delim=delim)
        assoc_string = assoc_string.format(delim=assoc_delim)
        with open(path, "w") as f:
            f.write(ipfstring)
        with open(path.parent.joinpath("A1000.txt"), "w") as f:
            f.write(assoc_string)
        with open(path.parent.joinpath("B2000.txt"), "w") as f:
            f.write(assoc_string)

    def teardown():
        remove("*.ipf")
        remove("*.txt")

    request.addfinalizer(teardown)
    return _write_assoc_ipf


@pytest.fixture(scope="module")
def nodata_ipf(request):
    df = pd.DataFrame()
    df["id"] = np.arange(3)
    df["nodatacolumn"] = np.nan
    df["thirdcolumn"] = "dummy"
    ipf.write("nodata.ipf", df)

    def teardown():
        remove("nodata.ipf")
    
    request.addfinalizer(teardown)


@pytest.fixture(scope="module")
def nodata_assoc(request):
    df = pd.DataFrame()
    df["id"] = np.arange(3)
    df["time"] = pd.date_range("2000-01-01", "2000-01-03")
    df["nodatacolumn"] = np.nan
    ipf.write_assoc("nodata.txt", df, itype="timeseries")

    def teardown():
        remove("nodata.txt")
    
    request.addfinalizer(teardown)


def test_read_associated__itype1implicit():
    path = "A1000.txt"
    delim = ","
    assoc_string = (
        "2\n"
        "2\n"
        "time{delim}-999.0\n"
        "level{delim}-999.0\n"
        "20180101000000{delim}1.0\n"
        "20180102000000{delim}-999.0\n"
    )
    with open(path, "w") as f:
        f.write(assoc_string.format(delim=delim))
    df = ipf.read_associated(path)
    assert df.shape == (2,2)

    delim = " "
    with open(path, "w") as f:
        f.write(assoc_string.format(delim=delim))
    df = ipf.read_associated(path, {"delim_whitespace": True})
    assert df.shape == (2,2)


def test_load__comma(write_basic_ipf):
    path = "basic_comma.ipf"
    write_basic_ipf(path, ",")
    df = ipf.load(path)
    assert isinstance(df, pd.DataFrame)
    assert list(df) == ["X", "Y", "Z", "City of Holland"]
    assert len(df) == 2
    assert df.iloc[0, 2] == -32.3
    assert df.iloc[1, 3] == "Den Bosch"


def test_load__space(write_basic_ipf):
    path = "basic_space.ipf"
    write_basic_ipf(path, " ")
    df = ipf.load(path, {"delim_whitespace": True})
    assert isinstance(df, pd.DataFrame)
    assert list(df) == ["X", "Y", "Z", "City of Holland"]
    assert len(df) == 2
    assert df.iloc[0, 2] == -32.3
    assert df.iloc[1, 3] == "Den Bosch"


def test_load_associated__comma_comma(write_assoc_ipf):
    path = "assoc.txt"
    write_assoc_ipf(path, ",", ",")
    df = ipf.load(path)

    nrecords, nfields = df.shape
    assert isinstance(df, pd.DataFrame)
    assert nrecords == 4
    assert nfields == 5
    assert df["time"].iloc[0] == pd.to_datetime("2018-01-01")
    assert df["time"].iloc[1] == pd.to_datetime("2018-01-02")
    assert df["level"].iloc[0] == df["level"].iloc[2] == 1.0
    assert pd.isnull(df["level"].iloc[1])
    assert pd.isnull(df["level"].iloc[3])


def test_load_associated__comma_space(write_assoc_ipf):
    path = "assoc.ipf"
    write_assoc_ipf(path, ",", " ")
    df = ipf.load(path, assoc_kwargs={"delim_whitespace": True})

    nrecords, nfields = df.shape
    assert isinstance(df, pd.DataFrame)
    assert nrecords == 4
    assert nfields == 5
    assert df["time"].iloc[0] == pd.to_datetime("2018-01-01")
    assert df["time"].iloc[1] == pd.to_datetime("2018-01-02")
    assert df["level"].iloc[0] == df["level"].iloc[2] == 1.0
    assert pd.isnull(df["level"].iloc[1])
    assert pd.isnull(df["level"].iloc[3])


def test_load_associated__space_space(write_assoc_ipf):
    path = "assoc.ipf"
    write_assoc_ipf(path, " ", " ")
    df = ipf.load(
        path, kwargs={"delim_whitespace": True}, assoc_kwargs={"delim_whitespace": True}
    )

    nrecords, nfields = df.shape
    assert isinstance(df, pd.DataFrame)
    assert nrecords == 4
    assert nfields == 5
    assert df["time"].iloc[0] == pd.to_datetime("2018-01-01")
    assert df["time"].iloc[1] == pd.to_datetime("2018-01-02")
    assert df["level"].iloc[0] == df["level"].iloc[2] == 1.0
    assert pd.isnull(df["level"].iloc[1])
    assert pd.isnull(df["level"].iloc[3])


def test_load_associated__space_comma(write_assoc_ipf):
    path = "assoc.ipf"
    write_assoc_ipf(path, " ", ",")
    df = ipf.load(path, kwargs={"delim_whitespace": True})

    nrecords, nfields = df.shape
    assert isinstance(df, pd.DataFrame)
    assert nrecords == 4
    assert nfields == 5
    assert df["time"].iloc[0] == pd.to_datetime("2018-01-01")
    assert df["time"].iloc[1] == pd.to_datetime("2018-01-02")
    assert df["level"].iloc[0] == df["level"].iloc[2] == 1.0
    assert pd.isnull(df["level"].iloc[1])
    assert pd.isnull(df["level"].iloc[3])


def test_write_assoc_itype1():
    times = [pd.to_datetime(s) for s in ["2018-01-01", "2018-02-01"]]
    df = pd.DataFrame.from_dict(
        {
            "x": [1, 1, 2, 2],
            "y": [3, 3, 4, 4],
            "id": ["A1", "A1", "B2", "B2"],
            "level": [0.1, 0.2, 0.3, 0.4],
            "time": times + times,
            "location": ["loc1", "loc1", "loc2", "loc2"],
        }
    )
    _, first_df = list(df.groupby("id"))[0]
    ipf.write_assoc("A1.txt", first_df, itype=1, nodata=-999.0)
    df2 = ipf.read_associated("A1.txt")
    pd.testing.assert_frame_equal(first_df, df2, check_like=True)

    remove("A1.txt")


def test_write_assoc_itype2():
    df = pd.DataFrame.from_dict(
        {
            "x": [1, 1, 2, 2],
            "y": [3, 3, 4, 4],
            "id": ["A1", "A1", "B2", "B2"],
            "litho": [0.1, np.nan, 0.3, np.nan],
            "top": [0.0, -0.5, -0.3, -0.5],
            "location": ["loc1", "loc1", "loc2", "loc2"],
        }
    )
    _, first_df = list(df.groupby("id"))[0]
    ipf.write_assoc("A1.txt", first_df, itype=2, nodata=-999.0)
    df2 = ipf.read_associated("A1.txt")
    pd.testing.assert_frame_equal(first_df, df2, check_like=True)

    remove("A1.txt")


def test_write():
    df = pd.DataFrame.from_dict(
        {
            "X": [100.0, 553.0],
            "Y": [435.0, 143.0],
            "/": [-32.3, -7.3],
            "City of Holland": ["Amsterdam", "Den Bosch"],
        }
    )
    ipf.write("basic.ipf", df)
    df2 = ipf.read("basic.ipf")
    pd.testing.assert_frame_equal(df, df2, check_like=True)

    remove("basic.ipf")


def test_lower_dataframe_colnames():
    colnames = ["X", "y", "iD"]
    out = ipf._lower(colnames)
    assert out == ["x", "y", "id"]


def test_lower_dataframe_colnames__ValueError():
    """Non-unique column names after lowering"""
    colnames = ["X", "y", "ID", "id"]
    with pytest.raises(ValueError):
        ipf._lower(colnames)


def test_is_single_value():
    df = pd.DataFrame(
        {
            "A": np.arange(8),
            "B": list("aabbbbcc"),
            "id": list("11112222"),
            "grp_const": [3, 3, 3, 3, 4, 4, 4, 4],
        }
    )
    grouped = df.groupby("id")
    assert not grouped["A"].apply(ipf._is_single_value).all()
    assert not grouped["B"].apply(ipf._is_single_value).all()
    assert grouped["grp_const"].apply(ipf._is_single_value).all()


def test_save__assoc_itype1():
    times = [pd.to_datetime(s) for s in ["2018-01-01", "2018-02-01"]]
    df = pd.DataFrame.from_dict(
        {
            "x": [1, 1, 2, 2],
            "y": [3, 3, 4, 4],
            "id": ["A1", "A1", "B2", "B2"],
            "level": [0.1, 0.2, 0.3, 0.4],
            "time": times + times,
            "location": ["loc1", "loc1", "loc2", "loc2"],
        }
    )

    ipf.save("save.ipf", df, itype=1, nodata=-999.0)
    assert Path("save.ipf").exists()
    assert Path("A1.txt").exists()
    assert Path("B2.txt").exists()
    df2 = ipf.load("save.ipf")
    df = df.sort_values(by="x")
    df2.index = df.index
    df2 = df2.sort_values(by="x")
    pd.testing.assert_frame_equal(df, df2, check_like=True)

    remove("save.ipf")
    remove("A1.txt")
    remove("B2.txt")


def test_save__assoc_itype2_():
    df = pd.DataFrame.from_dict(
        {
            "X": [1, 1, 2, 2],
            "Y": [3, 3, 4, 4],
            "ID": ["A1", "A1", "B2", "B2"],
            "litho": ["z", np.nan, "k", np.nan],
            "top": [0.0, -0.5, -0.3, -0.8],
            "location": ["loc1", "loc1", "loc2", "loc2"],
        }
    )

    ipf.save("save.ipf", df, itype=2, nodata=-999.0)
    assert Path("save.ipf").exists()
    assert Path("A1.txt").exists()
    assert Path("B2.txt").exists()
    df2 = ipf.load("save.ipf")
    df = df.sort_values(by="x")
    df2 = df2.sort_values(by="x")
    df2.index = df.index
    pd.testing.assert_frame_equal(df, df2, check_like=True)

    remove("save.ipf")
    remove("A1.txt")
    remove("B2.txt")


def test_save__assoc_itype1__layers():
    times = [pd.to_datetime(s) for s in ["2018-01-01", "2018-02-01"]]
    df = pd.DataFrame.from_dict(
        {
            "x": [1, 1, 2, 2, 3, 3, 4, 4],
            "y": [3, 3, 4, 4, 5, 5, 6, 6],
            "id": ["A1", "A1", "B2", "B2", "C3", "C3", "D4", "D4"],
            "level": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "time": times * 4,
            "location": [
                "loc1",
                "loc1",
                "loc2",
                "loc2",
                "loc3",
                "loc3",
                "loc4",
                "loc4",
            ],
            "layer": [1, 1, 1, 1, 3, 3, 3, 3],
        }
    )

    ipf.save("save.ipf", df, itype=1, nodata=-999.0)
    assert Path("save_l1.ipf").exists()
    assert Path("save_l3.ipf").exists()
    assert Path("A1.txt").exists()
    assert Path("B2.txt").exists()
    assert Path("C3.txt").exists()
    assert Path("D4.txt").exists()
    df2 = ipf.load("save_l*.ipf")
    df = df.sort_values(by="x")
    df2 = df2.sort_values(by="x")
    df2.index = df.index
    pd.testing.assert_frame_equal(df, df2, check_like=True)

    remove("save_l1.ipf")
    remove("save_l3.ipf")
    remove("A1.txt")
    remove("B2.txt")
    remove("C3.txt")
    remove("D4.txt")


def test_save__missing(nodata_ipf):
    """
    iMOD does not accept ",," for nodata. These should be filled in by a nodata
    values.
    """
    with open("nodata.ipf") as f:
        content = f.read()
    assert ",," not in content


def test_save__assoc_missing(nodata_assoc):
    """
    iMOD does not accept ",," for nodata. These should be filled in by a nodata
    values.
    """
    with open("nodata.txt") as f:
        content = f.read()
    assert ",," not in content