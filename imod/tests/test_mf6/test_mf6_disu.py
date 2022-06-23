import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

from imod.mf6 import disu


@pytest.fixture(scope="function")
def structured_dis():
    coords = {
        "layer": [1, 2, 3],
        "y": [25.0, 15.0, 5.0],
        "x": [50.0, 70.0, 90.0, 110.0],
    }
    dims = ["layer", "y", "x"]
    idomain = xr.DataArray(np.ones((3, 3, 4), dtype=np.int32), coords, dims)
    top = xr.DataArray(np.ones((3, 3, 4)), coords, dims)
    bottom = xr.DataArray(np.ones((3, 3, 4)), coords, dims)
    top.values[0] = 45.0
    top.values[1] = 30.0
    top.values[2] = 15.0
    bottom.values[0] = 30.0
    bottom.values[1] = 15.0
    bottom.values[2] = 0.0
    return idomain, top, bottom


def connectivity_checks(dis):
    ncell = dis.dataset["node"].size
    i = np.repeat(np.arange(1, ncell + 1), dis.dataset["iac"].values) - 1
    j = dis.dataset["ja"].values - 1
    diff = np.abs(i - j)
    ihc = dis.dataset["ihc"].values
    hwva = dis.dataset["hwva"].values
    cl12 = dis.dataset["cl12"].values

    # The node number itself is marked by a negative number.
    connection = j > 0
    vertical = (ihc == 0) & connection

    assert dis.dataset["iac"].values.sum() == j.size
    assert i.min() == 0
    assert i.max() == ncell - 1
    assert j.max() == ncell - 1
    assert (diff[connection] > 0).all()
    assert (cl12[connection] != 0).all()
    assert (hwva[connection] != 0).all()
    assert (diff[vertical] > 1).all()
    assert np.allclose(cl12[vertical], 7.5)
    assert np.allclose(hwva[vertical], 200.0)


def test_cell_number():
    nrow = 4
    ncol = 3
    assert disu._number(0, 0, 0, nrow, ncol) == 0
    assert disu._number(0, 0, 1, nrow, ncol) == 1
    assert disu._number(0, 1, 0, nrow, ncol) == 3
    assert disu._number(0, 1, 1, nrow, ncol) == 4
    assert disu._number(1, 0, 0, nrow, ncol) == 12
    assert disu._number(1, 0, 1, nrow, ncol) == 13
    assert disu._number(1, 1, 0, nrow, ncol) == 15
    assert disu._number(1, 1, 1, nrow, ncol) == 16


def test_structured_connectivity_full():
    idomain = np.ones((3, 3, 4), dtype=np.int32)
    i, j = disu._structured_connectivity(idomain)

    expected_i = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
    expected_j = np.array([1, 4, 12, 2, 5, 13, 3, 6, 14, 7, 15])
    assert np.array_equal(i[:11], expected_i)
    assert np.array_equal(j[:11], expected_j)

    idomain[1, 0, 1] = -1
    idomain[1, 0, 2] = -1
    i, j = disu._structured_connectivity(idomain)

    expected_i = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
    expected_j = np.array([1, 4, 12, 2, 5, 25, 3, 6, 26, 7, 15])
    assert np.array_equal(i[:11], expected_i)
    assert np.array_equal(j[:11], expected_j)


def test_from_structured(structured_dis):
    idomain, top, bottom = structured_dis
    dis = disu.LowLevelUnstructuredDiscretization.from_dis(top, bottom, idomain)
    assert np.allclose(dis.dataset["xorigin"], 40.0)
    assert np.allclose(dis.dataset["yorigin"], 0.0)
    connectivity_checks(dis)

    # Now disable some cells, create one pass-through
    idomain.values[1, 0, 1] = -1
    idomain.values[:, 0, 0] = 0
    dis = disu.LowLevelUnstructuredDiscretization.from_dis(top, bottom, idomain)
    connectivity_checks(dis)


def test_render(structured_dis):
    idomain, top, bottom = structured_dis
    dis = disu.LowLevelUnstructuredDiscretization.from_dis(top, bottom, idomain)

    directory = pathlib.Path("mymodel")
    actual = dis.render(directory, "disu", None, True)

    expected = textwrap.dedent(
        """\
        begin options
          xorigin 40.0
          yorigin 0.0
        end options

        begin dimensions
          nodes 36
          nja 186
        end dimensions

        begin griddata
          top
            open/close mymodel/disu/top.bin (binary)
          bot
            open/close mymodel/disu/bot.bin (binary)
          area
            open/close mymodel/disu/area.bin (binary)
          idomain
            open/close mymodel/disu/idomain.bin (binary)
        end griddata

        begin connectiondata
          iac
            open/close mymodel/disu/iac.bin (binary)
          ja
            open/close mymodel/disu/ja.bin (binary)
          ihc
            open/close mymodel/disu/ihc.bin (binary)
          cl12
            open/close mymodel/disu/cl12.bin (binary)
          hwva
            open/close mymodel/disu/hwva.bin (binary)
        end connectiondata"""
    )
    print(actual)
    print(expected)
    assert actual == expected


def test_write(structured_dis, tmp_path):
    idomain, top, bottom = structured_dis
    dis = disu.LowLevelUnstructuredDiscretization.from_dis(top, bottom, idomain)
    dis.write(tmp_path, "disu", None, True)

    assert (tmp_path / "disu.disu").exists
    names = [
        "top.bin",
        "bot.bin",
        "area.bin",
        "iac.bin",
        "ja.bin",
        "ihc.bin",
        "cl12.bin",
        "hwva.bin",
    ]
    for name in names:
        assert (tmp_path / name).exists
