import dask
import numpy as np
import pytest
import xarray as xr

import imod


@pytest.mark.usefixtures("twri_result")
def test_open_disgrb(twri_result):
    modeldir = twri_result
    with imod.util.cd(modeldir):
        grb = imod.mf6.out.open_disgrb("GWF_1/dis.dis.grb")
        assert isinstance(grb, dict)
        assert sorted(grb.keys()) == [
            "bottom",
            "coords",
            "ia",
            "icelltype",
            "idomain",
            "ja",
            "ncells",
            "ncol",
            "nja",
            "nlayer",
            "nrow",
            "top",
        ]
        assert isinstance(grb["bottom"], xr.DataArray)
        assert isinstance(grb["coords"], dict)
        assert isinstance(grb["ia"], np.ndarray)
        assert isinstance(grb["icelltype"], xr.DataArray)
        assert isinstance(grb["idomain"], xr.DataArray)
        assert isinstance(grb["ja"], np.ndarray)
        assert isinstance(grb["ncells"], int)
        assert isinstance(grb["ncol"], int)
        assert isinstance(grb["nja"], int)
        assert isinstance(grb["nlayer"], int)
        assert isinstance(grb["nrow"], int)
        assert isinstance(grb["top"], xr.DataArray)


@pytest.mark.usefixtures("twri_result", "twri_model")
def test_open_hds(twri_result, twri_model):
    model = twri_model["GWF_1"]
    modeldir = twri_result
    with imod.util.cd(modeldir):
        head = imod.mf6.open_hds("GWF_1/GWF_1.hds", "GWF_1/dis.dis.grb")
        assert head.dims == ("time", "layer", "y", "x")
        assert head.shape == (1, 3, 15, 15)
        assert np.allclose(head["time"].values, np.array([1.0]))
        assert np.allclose(head["x"].values, model["dis"]["x"].values)
        assert np.allclose(head["y"].values, model["dis"]["y"].values)
        assert isinstance(head.data, dask.array.Array)  # rather than numpy array


@pytest.mark.usefixtures("twri_result")
def test_read_cbc_headers(twri_result):
    modeldir = twri_result
    with imod.util.cd(modeldir):
        headers = imod.mf6.read_cbc_headers("GWF_1/GWF_1.cbc")
        assert isinstance(headers, dict)
        assert sorted(headers.keys()) == [
            "chd-1",
            "drn-1",
            "flow-ja-face",
            "wel-1",
        ]
        assert isinstance(headers["chd-1"], list)
        assert isinstance(headers["flow-ja-face"][0], imod.mf6.out.Imeth1Header)
        assert isinstance(headers["chd-1"][0], imod.mf6.out.Imeth6Header)


def test_dis_indices():
    # Note: ia and ja use 1-based indexing as produced by modflow6!
    # Cell numbering, top-view:
    #
    # +---+---+---+
    # | 1 | 2 | 3 |
    # +---+---+---+
    # | 4 | 5 | 6 |
    # +---+---+---+
    ia = np.array([1, 3, 6, 8, 10, 13, 15])
    # from       ([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6])
    ja = np.array([2, 4, 1, 3, 5, 2, 6, 1, 5, 2, 4, 6, 3, 5])
    # nja index    0  1  2  3  4  5  6  7  8  9 10  11 12 13
    ncells = 6
    nlayer = 1
    nrow = 2
    ncol = 3
    right, front, lower = imod.mf6.out._dis_indices(ia, ja, ncells, nlayer, nrow, ncol)
    assert right.shape == front.shape == lower.shape == (nlayer, nrow, ncol)
    assert (lower == -1).all()  # No lower connections
    right_expected = np.array(
        [
            [0, 3, -1],
            [8, 11, -1],
        ]
    )
    assert np.allclose(right[0], right_expected)
    front_expected = np.array(
        [
            [1, 4, 6],
            [-1, -1, -1],
        ]
    )
    assert np.allclose(front[0], front_expected)


def test_dis_indices__idomain():
    # Note: ia and ja use 1-based indexing as produced by modflow6!
    # Cell numbering, cross-section:
    #
    # +---+---+---+
    # | 1 | 2 | 3 |
    # +---+---+---+
    # | 4 | 5 | 6 |
    # +---+---+---+
    # | 7 | 8 | 9 |
    # +---+---+---+
    #
    # With idomain:
    #
    # +---+---+---+
    # | 1 | 1 | 1 |
    # +---+---+---+
    # | 1 |-1 | 0 |
    # +---+---+---+
    # | 1 | 1 | 1 |
    # +---+---+---+
    ia = np.array([1, 3, 6, 7, 9, 9, 9, 11, 14, 15])
    # ia index     1  2  3  4  5  6  7  8  9  10 11 12 13 14
    # from       ([1, 1, 2, 2, 2, 3, 4, 4, 7, 7, 8, 8, 8, 9)
    ja = np.array([2, 4, 1, 3, 8, 2, 1, 7, 4, 8, 7, 2, 9, 8])
    # nja index    0  1  2  3  4  5  6  7  8  9 10 11 12 13
    ncells = 9
    nlayer = 3
    nrow = 3
    ncol = 1
    right, front, lower = imod.mf6.out._dis_indices(ia, ja, ncells, nlayer, nrow, ncol)
    assert (front == -1).all()  # No front connections
    right_expected = np.array(
        [
            [0, 3, -1],
            [-1, -1, -1],
            [9, 12, -1],
        ]
    )
    assert np.allclose(right.reshape(nlayer, nrow), right_expected)
    lower_expected = np.array(
        [
            [1, 4, -1],
            [7, 4, -1],
            [-1, -1, -1],
        ]
    )
    assert np.allclose(lower.reshape(nlayer, nrow), lower_expected)


@pytest.mark.usefixtures("twri_result")
def test_open_cbc__dis(twri_result):
    modeldir = twri_result
    with imod.util.cd(modeldir):
        cbc = imod.mf6.open_cbc("GWF_1/GWF_1.cbc", "GWF_1/dis.dis.grb")
        assert isinstance(cbc, dict)
        assert sorted(cbc.keys()) == [
            "chd-1",
            "drn-1",
            "front-face-flow",
            "lower-face-flow",
            "right-face-flow",
            "wel-1",
        ]
        for array in cbc.values():
            assert array.shape == (1, 3, 15, 15)
            assert isinstance(array, xr.DataArray)
            assert isinstance(array.data, dask.array.Array)
