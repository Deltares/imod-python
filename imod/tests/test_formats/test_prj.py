import re
import textwrap
from datetime import datetime

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry as sg
import xarray as xr

import imod
from imod.formats.prj import prj


def test_tokenize():
    assert prj._tokenize("a b c") == ["a", "b", "c"]
    assert prj._tokenize("a,b,c") == ["a", "b", "c"]
    assert prj._tokenize("a, b, c") == ["a", "b", "c"]
    assert prj._tokenize("a, 'b', c") == ["a", "b", "c"]
    assert prj._tokenize("a, 'b d', c") == ["a", "b d", "c"]

    # We don't expect commas in our quoted strings since they're paths:
    with pytest.raises(ValueError, match="No closing quotation"):
        prj._tokenize("a, 'b,d', c")

    # From the examples:
    with pytest.raises(ValueError, match="No closing quotation"):
        prj._tokenize("That's life")
    assert prj._tokenize("That 's life'") == ["That", "s life"]
    assert prj._tokenize("That,'s life'") == ["That", "s life"]
    assert prj._tokenize("Thats life") == ["Thats", "life"]


class TestLineIterator:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.lines = prj._LineIterator(
            [
                ["This", "is", "the", "first", "line"],
                ["this", "is", "the", "second"],
            ]
        )

    def test_init(self):
        assert self.lines.count == -1

    def test_traversal(self):
        line = next(self.lines)
        assert line == ["This", "is", "the", "first", "line"]
        assert self.lines.count == 0
        line = next(self.lines)
        assert line == ["this", "is", "the", "second"]
        assert self.lines.count == 1
        assert self.lines.finished

        with pytest.raises(StopIteration):
            next(self.lines)

        assert self.lines.count == 1
        self.lines.back()
        assert self.lines.count == 0
        self.lines.back()
        assert self.lines.count == -1
        # Shouldn't go further back than -1
        self.lines.back()
        assert self.lines.count == -1

    def test_iter(self):
        lines = [line for line in self.lines]
        assert lines == [
            ["This", "is", "the", "first", "line"],
            ["this", "is", "the", "second"],
        ]


@pytest.mark.parametrize("error_type", [ValueError, TypeError])
def test_wrap_error_message(error_type):
    lines = prj._LineIterator([["one"], ["two"], ["three"]])
    exc = error_type("something wrong")
    expected = re.escape(
        "something wrong\n"
        "Failed to parse test content for line 1 with content:\n"
        "['one']"
    )
    with pytest.raises(error_type, match=expected):
        prj._wrap_error_message(exc, "test content", lines)


def test_parseblockheader():
    lines = prj._LineIterator(
        [
            ["abc", "def"],
            [],
        ]
    )
    assert prj._parse_blockheader(lines) == (None, None, None)
    assert prj._parse_blockheader(lines) == (None, None, None)

    lines = prj._LineIterator(
        [
            ["periods"],
            ["species"],
        ]
    )
    assert prj._parse_blockheader(lines) == (1, "periods", None)
    assert prj._parse_blockheader(lines) == (1, "species", None)

    lines = prj._LineIterator(
        [
            ["001", "(RIV)", "1"],
            ["002", "(GHB)", "0"],
            ["003", "(DRN)", "1", "extra", "content"],
        ]
    )
    assert prj._parse_blockheader(lines) == (1, "(riv)", "1")
    assert prj._parse_blockheader(lines) == (2, "(ghb)", "0")
    assert prj._parse_blockheader(lines) == (3, "(drn)", "1")

    # Test error wrapping
    lines = prj._LineIterator([["a", "b", "c"]])
    with pytest.raises(ValueError, match="Failed to parse block header"):
        prj._parse_blockheader(lines)


def test_parse_time():
    lines = prj._LineIterator(
        [
            ["steady-state"],
            ["2000-01-01"],
            ["2000-01-01", "12:01:02"],
        ]
    )
    assert prj._parse_time(lines) == "steady-state"
    assert prj._parse_time(lines) == "2000-01-01"
    assert prj._parse_time(lines) == "2000-01-01 12:01:02"

    lines = prj._LineIterator([1, 2, 3])
    with pytest.raises(TypeError, match="Failed to parse date time"):
        prj._parse_time(lines)


def test_parse_blockline():
    lines = prj._LineIterator(
        [
            ["1", "2", "001", "1.0", "0.0", "-999.99", "ibound.idf"],
            ["0", "2", "012", "2.0", "3.0", "-999.99", "ibound.idf"],
            ["1", "1", "012", "2.0", "3.0", "-999.99", "ibound.idf"],
        ]
    )
    actual = prj._parse_blockline(lines)
    expected = {
        "active": True,
        "is_constant": 2,
        "layer": 1,
        "factor": 1.0,
        "addition": 0.0,
        "constant": -999.99,
        "path": "ibound.idf",
    }
    assert actual == expected

    actual = prj._parse_blockline(lines, "2000-01-01 00:00:00")
    assert actual["time"] == "2000-01-01 00:00:00"
    assert not actual["active"]

    actual = prj._parse_blockline(lines, "2000-01-01 00:00:00")
    assert "path" not in actual

    lines = prj._LineIterator(
        [["1", "2", "001"]],
    )
    with pytest.raises(IndexError, match="Failed to parse entries"):
        prj._parse_blockline(lines)


def test_parse_nsub_nsystem():
    lines = prj._LineIterator([["5", "2"]])
    assert prj._parse_nsub_nsystem(lines) == (5, 2)

    lines = prj._LineIterator([["5"]])
    with pytest.raises(
        IndexError, match="Failed to parse number of sub-entries and number of systems"
    ):
        prj._parse_nsub_nsystem(lines)


def test_parse_notimeblock():
    fields = ["conductance", "head"]
    lines = prj._LineIterator(
        [
            ["3", "1"],
            ["a", "b", "c"],
        ]
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected NSUB entry of 2 for ['conductance', 'head'], read: 3"
        ),
    ):
        prj._parse_notimeblock(lines, fields)

    lines = prj._LineIterator(
        [
            ["2", "1"],
            ["1", "2", "001", "1.0", "0.0", "-999.99", "cond.idf"],
            ["1", "2", "001", "2.0", "3.0", "-999.99", "head.idf"],
        ]
    )
    actual = prj._parse_notimeblock(lines, fields)
    assert actual["n_system"] == 1
    assert actual["conductance"][0]["path"] == "cond.idf"
    assert actual["head"][0]["path"] == "head.idf"


def test_parse_capblock():
    lines = prj._LineIterator(
        [
            ["3", "1"],
            ["a", "b", "c"],
        ]
    )
    with pytest.raises(ValueError, match="Expected NSUB entry of 21, 22, or 26"):
        prj._parse_capblock(lines)


def test_parse_pcgblock():
    lines = prj._LineIterator(
        ["50,100,0.10000E-02,100.00,0.98000,1,1,0,1.0000,1.0000,0,0.10000".split(",")]
    )
    expected = {
        "mxiter": 50,
        "iter1": 100,
        "hclose": 1.0e-3,
        "rclose": 100.0,
        "relax": 0.98,
        "npcond": 1,
        "iprpcg": 1,
        "mutpcg": 0,
        "damppcg": 1.0,
        "damppcgt": 1.0,
        "iqerror": 0,
        "qerror": 0.1,
    }
    actual = prj._parse_pcgblock(lines)
    assert actual == expected

    lines = prj._LineIterator(
        # Try the different spacings...
        [
            ["mxiter=50"],
            ["iter1=", "100"],
            ["hclose", "=", "1.0e-3"],
            ["rclose=100.0"],
            ["relax=0.98"],
            ["npcond=1"],
            ["iprpcg=1"],
            ["mutpcg=0"],
            ["damppcg=1.0"],
            ["damppcgt=1.0"],
            ["iqerror=0"],
            ["qerror=0.1"],
        ]
    )
    actual = prj._parse_pcgblock(lines)
    assert actual == expected

    lines = prj._LineIterator(
        ["50,100,0.10000E-02,100.00,0.98000,1,1,0,1.0000,1.0000".split(",")]
    )
    with pytest.raises(ValueError, match="Failed to parse PCG entry"):
        prj._parse_pcgblock(lines)


def test__parse_block():
    lines = prj._LineIterator([["1", "(bla)", "1"]])
    content = {}
    with pytest.raises(
        ValueError, match=re.escape("Failed to recognize header keyword: (bla)")
    ):
        prj._parse_block(lines, content)


def test__parse_periodsblock():
    lines = prj._LineIterator(
        [
            ["summer"],
            ["01-04-1900", "00:00:00"],
            ["winter"],
            ["01-10-1900", "00:00:00"],
            [],
        ]
    )
    assert prj._parse_periodsblock(lines) == {
        "summer": "01-04-1900 00:00:00",
        "winter": "01-10-1900 00:00:00",
    }


def test_read_error(tmp_path):
    file_content = textwrap.dedent(
        """
        0001,(CAP),1
        022,001
        1,2, 001, 1.0, 0.0, -999.9900,"a.idf'
        """
    )
    path = tmp_path / "dummy.prj"
    with open(path, "w") as f:
        f.write(file_content)

    expected = re.escape("No closing quotation\nError occurred in line 3")
    with pytest.raises(ValueError, match=expected):
        prj.read_projectfile(path)


class TestProjectFile:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path_factory):
        # Try for maximum inconsistency...
        # Single and double quotes
        # General "comment-y" things after input, etc.
        # Empty line after section, empty line IN section, etc.
        # Technically project file paths should be absolute?...
        basepath = tmp_path_factory.mktemp("prj-data")
        self.basepath = basepath

        filecontent = textwrap.dedent(
            f"""
            0001,(CAP),1,MetaSwap,[bnd,lus,rtz,slt,mst,sfl,rti,rti,rti,wra,wua,pua,pra,rua,rra,rua,rra,iua,ira,pwd,smf,spf]
            022,001
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf"
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf"
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf"
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf"
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf"
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf",
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf",
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf",
            1,1, 001, 1.0, 0.0,  25.00000,''  # capacity
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf"
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf"
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf"
            1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf"
            1,1, 001, 1.0, 0.0,  1.0     ,''  # runof resistance urban area
            1,1, 001, 1.0, 0.0,  1.0     ,''  # runof resistance rural area
            1,1, 001, 1.0, 0.0,  1.0     ,''  # runon resistance urban area
            1,1, 001, 1.0, 0.0,  1.0     ,''  # runon resistance rural area
            1,1, 001, 1.0, 0.0,  0.0     ,''  # qinfbasic urban area
            1,1, 001, 1.0, 0.0,  0.05    ,''  # qinfbasic urban area
            1,1, 001, 1.0, 0.0,  0.0     ,''  # pwt level
            1,1, 001, 1.0, 0.0,  1.0     ,''  # soil moisture factor
            1,1, 001, 1.0, 0.0,  1.0     ,''  # conductivity factor
            008,EXTRA FILES
            aa.idf
            ab.idf
            ac.idf
            ad.idf
            ae.idf
            af.idf
            ag.idf
            ah.idf

            0001,(BND),1, Boundary Condition,[BND]
            001,002
             1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/a.idf",
             1,2, 002, 1.0, 0.0, -999.9900,"{basepath}/a.idf",
            0001,(KHV),1, Horizontal Permeability,[KHV]
            001,002
             1,2, 001, 1.0, 0.0, -999.9900,'{basepath}/a.idf',
             1,2, 002, 1.0, 0.0, -999.9900,"{basepath}/a.idf",

            0003,(GHB),1, General Head Boundary,[GHB]
            1900-01-01
            002,001
             1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/cond.idf",
             1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/head.idf",
            1901-01-02 00:00:00
            002,001
             1,2, 001, 1.0, 0.0, -999.9900,'{basepath}/cond.idf',
             1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/head.idf",
            1901-01-03 00:00:00
            002,001
             1,2, 001, 1.0, 0.0, -999.9900,'{basepath}/cond.idf',
             1,2, 001, 1.0, 0.0, -999.9900,"{basepath}/head.idf",

            0001,(WEL),1, Wells,[WRA]
            STEADY-STATE
            001,002
             1,2, 001, 1.0, 0.0, -999.9900 ,"{basepath}/wells_l1.ipf"
             1,2, 002, 1.0, 0.0, -999.9900 ,"{basepath}/wells_l2.ipf"

            0001,(HFB),1, Horizontal Flow Barrier,[HFB]
            001,002
             1,2, 000, 100000.0, 1.0, -999.9900 ,"{basepath}/first.gen"
             1,2, 000, 100000.0, 1.0, -999.9900 ,"{basepath}/second.gen"

            0001,(PCG),1, Precondition Conjugate-Gradient,[]
             MXITER=  500
             ITER1=   25
             HCLOSE=  0.1000000E-02
             RCLOSE=  0.1000000
             RELAX=   0.9800000
             NPCOND=  1
             IPRPCG=  1
             MUTPCG=  0
             DAMPPCG= 1.000000
             DAMPPCGT=1.000000
             IQERROR= 0
             QERROR=  0.1000000


            Periods
            summer
            01-04-1900 00:00:00
            winter
            01-10-1900 00:00:00

            """
        )

        self.prj_path = basepath / "testprojectfile.prj"
        with open(self.prj_path, "w") as f:
            f.write(filecontent)

        # Write first GEN as binary
        geom = sg.LineString(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=[geom, geom])
        imod.gen.write(basepath / "first.gen", gdf)

        # Write second GEN as ASCII
        gen_content = textwrap.dedent(
            """
        1
          1.0, 0.0
          1.0, 1.0
          2.0, 0.0
        END
        END
        """
        )
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=[geom, geom])
        with open(basepath / "second.gen", "w") as f:
            f.write(gen_content)

        da = xr.DataArray(
            data=[[0.0, 1.0], [2.0, 3.0]],
            coords={"x": [0.5, 1.5], "y": [3.5, 2.5]},
            dims=["y", "x"],
        )
        imod.idf.save(basepath / "a", da)
        imod.idf.save(basepath / "cond", da)
        imod.idf.save(basepath / "head", da)

        df = pd.DataFrame(
            data={
                "id": [1, 2, 3],
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0],
                "rate": [2.0, 3.0, 4.0],
            }
        )
        imod.ipf.save(basepath / "wells_l1.ipf", df)
        imod.ipf.save(basepath / "wells_l2.ipf", df)

    def test_read_projectfile(self):
        content = imod.prj.read_projectfile(self.prj_path)
        assert isinstance(content, dict)

    def test_open_projectfile_data(self):
        content, repeats = imod.prj.open_projectfile_data(self.prj_path)
        assert isinstance(content, dict)
        assert isinstance(repeats, dict)
        assert isinstance(content["ghb"]["conductance"], xr.DataArray)
        assert isinstance(content["ghb"]["head"], xr.DataArray)
        assert isinstance(content["cap"]["landuse"], xr.DataArray)
        assert isinstance(content["wel-1"]["dataframe"], pd.DataFrame)
        assert isinstance(content["wel-2"]["dataframe"], pd.DataFrame)
        assert isinstance(content["hfb-1"]["geodataframe"], gpd.GeoDataFrame)
        assert isinstance(content["hfb-2"]["geodataframe"], gpd.GeoDataFrame)
        assert isinstance(content["pcg"], dict)
        assert content["periods"] == {
            "summer": datetime(1900, 4, 1),
            "winter": datetime(1900, 10, 1),
        }
