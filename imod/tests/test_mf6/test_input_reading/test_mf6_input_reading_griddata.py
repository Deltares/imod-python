import dask.array
import numpy as np
import pytest

from imod.mf6.read_input import grid_data as gd


def test_advance_to_griddata_section(tmp_path):
    path = tmp_path / "griddata.txt"
    content = "\n".join(
        [
            "",
            " idomain",
            "   open/close GWF_1/dis/idomain.dat",
            " botm layered",
            "",
        ]
    )

    with open(path, "w") as f:
        f.write(content)

    with open(path) as f:
        assert gd.advance_to_griddata_section(f) == ("idomain", False)
        f.readline()
        assert gd.advance_to_griddata_section(f) == ("botm", True)
        with pytest.raises(ValueError, match="No end of griddata specified"):
            gd.advance_to_griddata_section(f)


def test_shape_to_max_rows():
    assert gd.shape_to_max_rows(shape=(4, 3, 2), layered=False) == (12, (4, 3, 2))
    assert gd.shape_to_max_rows(shape=(4, 3, 2), layered=True) == (3, (3, 2))
    assert gd.shape_to_max_rows(shape=(3, 2), layered=False) == (3, (3, 2))
    assert gd.shape_to_max_rows(shape=(3, 2), layered=True) == (1, (2,))
    assert gd.shape_to_max_rows(shape=(6,), layered=False) == (1, (6,))
    with pytest.raises(
        ValueError, match="LAYERED section detected. DISU does not support LAYERED"
    ):
        gd.shape_to_max_rows(shape=(6,), layered=True)
    with pytest.raises(ValueError, match="length of shape should be 1, 2, or 3"):
        gd.shape_to_max_rows(shape=(5, 4, 3, 2), layered=True)


def test_constant():
    shape = (3, 4, 5)
    a = gd.constant(2.0, shape, np.float64)
    assert isinstance(a, dask.array.Array)
    b = a.compute()
    assert np.allclose(b, 2.0)


def test_read_internal_griddata(tmp_path):
    path = tmp_path / "internal.txt"
    with open(path, "w") as f:
        f.write("1 2 3 4\n5 6 7 8")

    with open(path) as f:
        a = gd.read_internal_griddata(f, np.int32, (2, 4), 2)

    assert a.shape == (2, 4)
    assert np.array_equal(a.ravel(), np.arange(1, 9))


def test_read_external_griddata(tmp_path):
    path1 = tmp_path / "external.dat"
    path2 = tmp_path / "external.bin"

    a = np.ones((4, 2))
    a.tofile(path1, sep=" ")
    a.tofile(path2)  # binary

    b = gd.read_external_griddata(path1, np.float64, (4, 2), False)
    c = gd.read_external_griddata(path2, np.float64, (4, 2), True)
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_read_array(tmp_path):
    blockfile_path = tmp_path / "blockfile.txt"
    external_path = tmp_path / "external.dat"
    external_binary_path = tmp_path / "external.bin"

    content = "\n".join(
        [
            "top",
            "  constant 200.0",
            "idomain",
            "  open/close external.dat",
            "botm",
            "  open/close external.bin (binary)",
            "abc",
        ]
    )
    with open(blockfile_path, "w") as f:
        f.write(content)

    shape = (3, 4, 5)
    a = np.ones(shape, dtype=np.float64)
    a.tofile(external_path, sep=" ")
    a.tofile(external_binary_path)

    with open(blockfile_path) as f:
        f.readline()
        top = gd.read_array(f, tmp_path, np.float64, max_rows=12, shape=shape)
        f.readline()
        idomain = gd.read_array(f, tmp_path, np.float64, max_rows=12, shape=shape)
        f.readline()
        botm = gd.read_array(f, tmp_path, np.float64, max_rows=12, shape=shape)
        with pytest.raises(
            ValueError, match='Expected "constant", "internal", or "open/close"'
        ):
            gd.read_array(f, tmp_path, np.float64, max_rows=12, shape=shape)

    for a in (top, idomain, botm):
        assert isinstance(a, dask.array.Array)
        assert a.shape == shape

    assert np.allclose(top, 200.0)
    assert np.allclose(idomain, 1)
    assert np.allclose(botm, 1.0)


def test_read_griddata(tmp_path):
    blockfile_path = tmp_path / "blockfile.txt"
    external_path = tmp_path / "external.dat"
    external_binary_path = tmp_path / "external.bin"

    content = "\n".join(
        [
            "top",
            "  constant 200.0",
            "idomain",
            "  open/close external.dat",
            "botm",
            "  open/close external.bin (binary)",
            "end",
        ]
    )
    with open(blockfile_path, "w") as f:
        f.write(content)

    sections = {
        "top": (np.float64, gd.shape_to_max_rows),
        "idomain": (np.int32, gd.shape_to_max_rows),
        "botm": (np.float64, gd.shape_to_max_rows),
    }
    shape = (3, 4, 5)
    a = np.ones(shape, dtype=np.int32)
    a.tofile(external_path, sep=" ")
    b = np.ones(shape, dtype=np.float64)
    b.tofile(external_binary_path)

    with open(blockfile_path) as f:
        d = gd.read_griddata(f, tmp_path, sections, shape)

    for key in ("top", "idomain", "botm"):
        assert key in d
        a = d[key]
        assert isinstance(a, dask.array.Array)
        assert a.shape == shape

    assert np.allclose(d["top"], 200.0)
    assert np.allclose(d["idomain"], 1)
    assert np.allclose(d["botm"], 1.0)

    dummy_path = tmp_path / "dummy.txt"
    with open(dummy_path, "w") as f:
        f.write("\n")

    with open(dummy_path) as f:
        with pytest.raises(ValueError, match="Error reading"):
            d = gd.read_griddata(f, tmp_path, sections, shape)
