import numpy as np
import pytest

from imod.mf6.read_input import common as cm


def isa(value, kind, expected):
    return isinstance(value, kind) and value == expected


def test_end_of_file():
    assert cm.end_of_file("") is True
    assert cm.end_of_file(" ") is False
    assert cm.end_of_file("\n") is False


def test_strip_line():
    assert cm.strip_line("abc") == "abc"
    assert cm.strip_line("abc ") == "abc"
    assert cm.strip_line("ABC ") == "abc"
    assert cm.strip_line("abc#def") == "abc"
    assert cm.strip_line("abc #def") == "abc"
    assert cm.strip_line(" abc ##def") == "abc"


def test_flatten():
    assert cm.flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert cm.flatten([["abc", "def"], ["ghi"]]) == ["abc", "def", "ghi"]


def test_split_line():
    assert cm.split_line("abc def ghi") == ["abc", "def", "ghi"]
    assert cm.split_line("abc,def,ghi") == ["abc", "def", "ghi"]
    assert cm.split_line("abc,def, ghi") == ["abc", "def", "ghi"]
    assert cm.split_line("abc ,def, ghi ") == ["abc", "def", "ghi"]


def test_to_float():
    assert isa(cm.to_float("1"), float, 1.0)
    assert isa(cm.to_float("1.0"), float, 1.0)
    assert isa(cm.to_float("1.0d0"), float, 1.0)
    assert isa(cm.to_float("1.0e0"), float, 1.0)
    assert isa(cm.to_float("1.0+0"), float, 1.0)
    assert isa(cm.to_float("1.0-0"), float, 1.0)
    assert isa(cm.to_float("1.0e-0"), float, 1.0)
    assert isa(cm.to_float("1.0e+0"), float, 1.0)


def test_find_entry():
    assert isa(cm.find_entry("abc factor 1", "factor", float), float, 1.0)
    assert isa(cm.find_entry("abc factor 1", "factor", int), int, 1)
    assert isa(cm.find_entry("abc factor 1", "factor", str), str, "1")
    assert cm.find_entry("abc 1", "factor", float) is None


def test_read_internal(tmp_path):
    path1 = tmp_path / "internal-1.dat"
    with open(path1, "w") as f:
        f.write("1 2 3 4")
    # max_rows should read all lines, unless max_rows is exceeded.
    with open(path1) as f:
        a = cm.read_internal(f, int, 1)
    with open(path1) as f:
        b = cm.read_internal(f, int, 2)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.array_equal(a, b)

    path2 = tmp_path / "internal-2.dat"
    with open(path2, "w") as f:
        f.write("1 2\n3 4")
    with open(path2) as f:
        a = cm.read_internal(f, float, 1)
    with open(path2) as f:
        b = cm.read_internal(f, int, 2)
    assert np.issubdtype(a.dtype, np.floating)
    assert np.issubdtype(b.dtype, np.integer)
    assert a.size == 2
    assert b.size == 4


def test_read_external_binaryfile(tmp_path):
    path1 = tmp_path / "external-1.bin"
    a = np.ones((5, 5))
    a.tofile(path1)
    b = cm.read_external_binaryfile(path1, np.float64, 25)
    assert b.shape == (25,)
    b = cm.read_external_binaryfile(path1, np.float64, 10)
    assert b.shape == (10,)

    dtype = np.dtype([("node", np.int32), ("value", np.float32)])
    a = np.ones((5,), dtype)
    path2 = tmp_path / "external-2.bin"
    a.tofile(path2)
    b = cm.read_external_binaryfile(path2, dtype, 5)
    assert np.array_equal(a, b)


def test_read_fortran_deflated_text_array(tmp_path):
    path1 = tmp_path / "deflated-1.txt"
    with open(path1, "w") as f:
        f.write("1.0\n2*2.0\n3*3.0")
    a = cm.read_fortran_deflated_text_array(path1, float, 3)
    b = np.array([1.0, 2.0, 2.0])
    assert np.array_equal(a, b)

    a = cm.read_fortran_deflated_text_array(path1, float, 6)
    b = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    assert np.array_equal(a, b)


def test_read_external_textfile(tmp_path):
    path1 = tmp_path / "external.dat"
    with open(path1, "w") as f:
        f.write("1.0 2.0 3.0")
    a = cm.read_external_textfile(path1, float, 6)
    b = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(a, b)

    path2 = tmp_path / "deflated-1.txt"
    with open(path2, "w") as f:
        f.write("1.0\n2*2.0\n3*3.0")
    a = cm.read_external_textfile(path2, float, 6)
    b = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    assert np.array_equal(a, b)


def test_advance_to_header(tmp_path):
    path = tmp_path / "header.txt"
    content = "\n".join(
        [
            "",
            "begin options",
            "1",
            "end options\n",
            "\n",
            "begin griddata",
            "2",
            "end griddata",
        ]
    )
    with open(path, "w") as f:
        f.write(content)

    with open(path) as f:
        cm.advance_to_header(f, "options")
        assert f.readline() == "1\n"
        f.readline()  # read "end options" line
        cm.advance_to_header(f, "griddata")
        assert f.readline() == "2\n"

        with pytest.raises(ValueError, match='"begin perioddata" is not present'):
            cm.advance_to_header(f, "perioddata")


def test_parse_option():
    with pytest.raises(ValueError, match="Cannot parse option in options.txt"):
        cm.parse_option("", "options.txt")
    assert cm.parse_option("save_flows", "options.txt") == ("save_flows", True)
    assert cm.parse_option("variablecv dewatered", "options.txt") == (
        "variablecv",
        "dewatered",
    )
    assert cm.parse_option("multiple a b c", "options.txt") == (
        "multiple",
        ["a", "b", "c"],
    )


def test_read_key_value_block(tmp_path):
    path = tmp_path / "values.txt"
    content = "\n".join(["", "save_flows", "variablecv dewatered", "", "end options"])

    with open(path, "w") as f:
        f.write(content)

    with open(path) as f:
        d = cm.read_key_value_block(f, cm.parse_option)
    assert d == {"save_flows": True, "variablecv": "dewatered"}

    path2 = tmp_path / "values-unterminated.txt"
    content = "\n".join(
        [
            "",
            "save_flows",
            "variablecv dewatered",
            "",
        ]
    )

    with open(path2, "w") as f:
        f.write(content)

    with open(path2) as f:
        with pytest.raises(ValueError, match='"end" of block is not present in file'):
            cm.read_key_value_block(f, cm.parse_option)


def test_read_iterable_block(tmp_path):
    def parse(line, _):  # second arg is normally file name
        return line.split()

    path = tmp_path / "iterable-values.txt"
    content = "\n".join(["1.0 1 1.0", "1.0 1 1.0", "1.0 1 1.0", "end perioddata"])
    with open(path, "w") as f:
        f.write(content)

    with open(path) as f:
        back = cm.read_iterable_block(f, parse)
    assert back == [
        ["1.0", "1", "1.0"],
        ["1.0", "1", "1.0"],
        ["1.0", "1", "1.0"],
    ]

    path2 = tmp_path / "iterable-values-unterminated.txt"
    content = "\n".join(
        [
            "1.0 1 1.0",
            "1.0 1 1.0",
            "1.0 1 1.0",
        ]
    )
    with open(path2, "w") as f:
        f.write(content)
    with open(path2) as f:
        with pytest.raises(ValueError, match='"end" of block is not present in file'):
            back = cm.read_iterable_block(f, parse)


def test_parse_dimension():
    assert cm.parse_dimension("  nper 3  ", "fname") == ("nper", 3)


def test_advance_to_period(tmp_path):
    path = tmp_path / "period.txt"
    content = "\n".join(
        [
            "" "  begin period 1",
            "2",
        ]
    )
    with open(path, "w") as f:
        f.write(content)

    with open(path) as f:
        cm.advance_to_period(f)
        assert f.readline() == "2"
