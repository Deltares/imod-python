import glob
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from imod import ipf


@pytest.fixture(scope="module")
def write_basic_ipf():

    # factory function seems easiest way to parameterize tests
    def _write_basic_ipf(path, delim):
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

    return _write_basic_ipf


@pytest.fixture(scope="module")
def write_assoc_ipf():
    def _write_assoc_ipf(path, delim1, delim2, delim3, delim4, delim5):
        ipfstring = (
            "2\n"
            "3\n"
            "X\n"
            "Y\n"
            "ID\n"
            "3{delim1}txt\n"
            "100.0{delim2}435.0{delim2}A1000\n"
            "553.0{delim2}143.0{delim2}B2000\n"
        )
        assoc_string = (
            "2\n"
            "2{delim3}1\n"
            "time{delim4}-999.0\n"
            "level{delim4}-999.0\n"
            "20180101000000{delim5}1.0\n"
            "20180102000000{delim5}-999.0\n"
        )
        ipfstring = ipfstring.format(delim1=delim1, delim2=delim2)
        assoc_string = assoc_string.format(delim3=delim3, delim4=delim4, delim5=delim5)
        with open(path, "w") as f:
            f.write(ipfstring)
        with open(path.parent / "A1000.txt", "w") as f:
            f.write(assoc_string)
        with open(path.parent / "B2000.txt", "w") as f:
            f.write(assoc_string)

    return _write_assoc_ipf


@pytest.fixture(scope="module")
def nodata_ipf(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("nodata_ipf")
    df = pd.DataFrame()
    df["id"] = np.arange(3)
    df["nodatacolumn"] = np.nan
    df["thirdcolumn"] = "dummy"
    ipf.write(tmp_dir / "nodata.ipf", df)
    return tmp_dir / "nodata.ipf"


@pytest.fixture(scope="module")
def nodata_assoc(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("nodata_ipf")
    df = pd.DataFrame()
    df["id"] = np.arange(3)
    df["time"] = pd.date_range("2000-01-01", "2000-01-03")
    df["nodatacolumn"] = np.nan
    ipf.write_assoc(tmp_dir / "nodata.txt", df, itype="timeseries")
    return tmp_dir / "nodata.txt"


def test_read_associated__itype1implicit(tmp_path):
    path = tmp_path / "A1000.txt"
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
    assert df.shape == (2, 2)

    delim = " "
    with open(path, "w") as f:
        f.write(assoc_string.format(delim=delim))
    df = ipf.read_associated(path, {"delim_whitespace": True})
    assert df.shape == (2, 2)


def test_read__comma(write_basic_ipf, tmp_path):
    path = tmp_path / "basic_comma.ipf"
    write_basic_ipf(path, ",")
    df = ipf.read(path)
    assert isinstance(df, pd.DataFrame)
    assert list(df) == ["X", "Y", "Z", "City of Holland"]
    assert len(df) == 2
    assert df.iloc[0, 2] == -32.3
    assert df.iloc[1, 3] == "Den Bosch"


def test_read__space(write_basic_ipf, tmp_path):
    path = tmp_path / "basic_space.ipf"
    write_basic_ipf(path, " ")
    df = ipf.read(path, {"delim_whitespace": True})
    assert isinstance(df, pd.DataFrame)
    assert list(df) == ["X", "Y", "Z", "City of Holland"]
    assert len(df) == 2
    assert df.iloc[0, 2] == -32.3
    assert df.iloc[1, 3] == "Den Bosch"


@pytest.mark.parametrize("delim1", [",", " "])
@pytest.mark.parametrize("delim2", [",", " "])
@pytest.mark.parametrize("delim3", [",", " "])
@pytest.mark.parametrize("delim4", [",", " "])
@pytest.mark.parametrize("delim5", [",", " "])
def test_read_associated__parameterized_delim(
    write_assoc_ipf, delim1, delim2, delim3, delim4, delim5, tmp_path
):
    path = tmp_path / "assoc.txt"
    write_assoc_ipf(path, delim1, delim2, delim3, delim4, delim5)
    df = ipf.read(path)

    nrecords, nfields = df.shape
    assert isinstance(df, pd.DataFrame)
    assert nrecords == 4
    assert nfields == 5
    assert df["time"].iloc[0] == pd.to_datetime("2018-01-01")
    assert df["time"].iloc[1] == pd.to_datetime("2018-01-02")
    assert df["level"].iloc[0] == df["level"].iloc[2] == 1.0
    assert pd.isnull(df["level"].iloc[1])
    assert pd.isnull(df["level"].iloc[3])


def test_write_assoc_itype1(tmp_path):
    times = [pd.to_datetime(s) for s in ["2018-01-01", "2018-02-01"]]
    df = pd.DataFrame.from_dict(
        {
            "x": [1, 1, 2, 2],
            "y": [3, 3, 4, 4],
            "id": ["A1", "A1", "B2", "B2"],
            "level": [0.1, 0.2, 0.3, 0.4],
            "time": times + times,
            "location": ["loc one", "loc one", "loc two", "loc two"],
        }
    )
    _, first_df = list(df.groupby("id"))[0]
    ipf.write_assoc(tmp_path / "A1.txt", first_df, itype=1, nodata=-999.0)
    df2 = ipf.read_associated(tmp_path / "A1.txt")
    pd.testing.assert_frame_equal(first_df, df2, check_like=True)
    # check quoting, to prevent imod separator confusion
    with open(tmp_path / "A1.txt") as io:
        lastline = io.readlines()[-1].rstrip()
    assert lastline == '20180201000000,1,3,"A1",0.2,"loc one"'


def test_write_assoc_itype2(tmp_path):
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
    ipf.write_assoc(tmp_path / "A1.txt", first_df, itype=2, nodata=-999.0)
    df2 = ipf.read_associated(tmp_path / "A1.txt")
    pd.testing.assert_frame_equal(first_df, df2, check_like=True)


def test_write(tmp_path):
    df = pd.DataFrame.from_dict(
        {
            "X": [100.0, 553.0],
            "Y": [435.0, 143.0],
            "/": [-32.3, -7.3],
            "City of Holland": ["Amsterdam", "Den Bosch"],
        }
    )
    ipf.write(tmp_path / "basic.ipf", df)
    df2 = ipf.read(tmp_path / "basic.ipf")
    pd.testing.assert_frame_equal(df, df2, check_like=True)


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


def test_save__assoc_itype1(tmp_path):
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

    ipf.save(tmp_path / "save.ipf", df, itype=1, nodata=-999.0)
    assert (tmp_path / "save.ipf").exists()
    assert (tmp_path / "A1.txt").exists()
    assert (tmp_path / "B2.txt").exists()
    df2 = ipf.read(tmp_path / "save.ipf")
    df = df.sort_values(by="x")
    df2.index = df.index
    df2 = df2.sort_values(by="x")
    pd.testing.assert_frame_equal(df, df2, check_like=True)


def test_save__assoc_itype2_(tmp_path):
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

    ipf.save(tmp_path / "save.ipf", df, itype=2, nodata=-999.0)
    assert (tmp_path / "save.ipf").exists()
    assert (tmp_path / "A1.txt").exists()
    assert (tmp_path / "B2.txt").exists()
    df2 = ipf.read(tmp_path / "save.ipf")
    df = df.sort_values(by="x")
    df2 = df2.sort_values(by="x")
    df2.index = df.index
    pd.testing.assert_frame_equal(df, df2, check_like=True)


def test_save__assoc_itype1__layers(tmp_path):
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

    ipf.save(tmp_path / "save.ipf", df, itype=1, nodata=-999.0)
    assert (tmp_path / "save_l1.ipf").exists()
    assert (tmp_path / "save_l3.ipf").exists()
    assert (tmp_path / "A1.txt").exists()
    assert (tmp_path / "B2.txt").exists()
    assert (tmp_path / "C3.txt").exists()
    assert (tmp_path / "D4.txt").exists()
    df2 = ipf.read(tmp_path / "save_l*.ipf")
    df = df.sort_values(by="x")
    df2 = df2.sort_values(by="x")
    df2.index = df.index
    pd.testing.assert_frame_equal(df, df2, check_like=True)


def test_save__missing(nodata_ipf):
    """
    iMOD does not accept ",," for nodata. These should be filled in by a nodata
    values.
    """
    with open(nodata_ipf) as f:
        content = f.read()
    assert ",," not in content


def test_error_fileinfo(tmp_path):
    """
    Test whether the filename is given in the error.
    """
    path = tmp_path / "good.ipf"
    ipfstring = (
        "2\n"
        "3\n"
        "X\n"
        "Y\n"
        "ID\n"
        "3{delim1}txt\n"
        "100.0{delim2}435.0{delim2}bad1\n"
        "553.0{delim2}143.0{delim2}bad2\n"
    )
    # Wrong number or entries in last row
    bad_assoc_string = (
        "2\n"
        "2{delim3}1\n"
        "time{delim4}-999.0\n"
        "level{delim4}-999.0\n"
        "20180101000000{delim5}1.0\n"
        '20180102000000{delim5}"fill\t,1.0\n'
    )
    delim1 = delim2 = delim3 = delim4 = delim5 = ","
    ipfstring = ipfstring.format(delim1=delim1, delim2=delim2)
    bad_assoc_string = bad_assoc_string.format(
        delim3=delim3, delim4=delim4, delim5=delim5
    )
    with open(path, "w") as f:
        f.write(ipfstring)
    with open(path.parent / "bad1.txt", "w") as f:
        f.write(bad_assoc_string)
    with open(path.parent / "bad2.txt", "w") as f:
        f.write(bad_assoc_string)

    with pytest.raises(
        # Match uses regex, and the base path of the IPF is non-deterministic
        # due to tmp_path
        pd.errors.ParserError,
        match=r'".*bad1\.txt" of IPF file ".*good.ipf"',
    ):
        ipf.read(path)


def test_save__assoc_itype1__layers__integerID(tmp_path):
    times = [pd.to_datetime(s) for s in ["2018-01-01", "2018-02-01"]]
    df = pd.DataFrame.from_dict(
        {
            "x": [1, 1, 2, 2, 3, 3, 4, 4],
            "y": [3, 3, 4, 4, 5, 5, 6, 6],
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
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

    ipf.save(tmp_path / "save.ipf", df, itype=1, nodata=-999.0)
    assert (tmp_path / "save_l1.ipf").exists()
    assert (tmp_path / "save_l3.ipf").exists()
    assert (tmp_path / "1.txt").exists()
    assert (tmp_path / "2.txt").exists()
    assert (tmp_path / "3.txt").exists()
    assert (tmp_path / "4.txt").exists()
