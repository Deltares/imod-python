import dask
import numpy as np
import xarray as xr
import xugrid as xu

import imod


def test_read_disgrb(twri_result):
    modeldir = twri_result
    with imod.util.cd(modeldir):
        grb = imod.mf6.out.read_grb("GWF_1/dis.dis.grb")
        assert isinstance(grb, dict)
        assert sorted(grb.keys()) == [
            "bottom",
            "coords",
            "distype",
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
        assert grb["distype"] == "dis"
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


def test_read_cbc_headers(twri_result):
    modeldir = twri_result
    with imod.util.cd(modeldir):
        headers = imod.mf6.read_cbc_headers("GWF_1/GWF_1.cbc")
        assert isinstance(headers, dict)
        assert sorted(headers.keys()) == [
            "chd_chd",
            "drn_drn",
            "flow-ja-face",
            "wel_wel",
        ]
        assert isinstance(headers["chd_chd"], list)
        assert isinstance(headers["flow-ja-face"][0], imod.mf6.out.cbc.Imeth1Header)
        assert isinstance(headers["chd_chd"][0], imod.mf6.out.cbc.Imeth6Header)


def test_read_cbc_headers__transient(transient_twri_result):
    modeldir = transient_twri_result
    with imod.util.cd(modeldir):
        headers = imod.mf6.read_cbc_headers("GWF_1/GWF_1.cbc")
        assert isinstance(headers, dict)
        assert sorted(headers.keys()) == [
            "chd_chd",
            "drn_drn",
            "flow-ja-face",
            "sto-ss",
            "wel_wel",
        ]
        assert isinstance(headers["chd_chd"], list)
        assert isinstance(headers["flow-ja-face"][0], imod.mf6.out.cbc.Imeth1Header)
        assert isinstance(headers["chd_chd"][0], imod.mf6.out.cbc.Imeth6Header)
        assert isinstance(headers["sto-ss"][0], imod.mf6.out.cbc.Imeth1Header)


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
    right, front, lower = imod.mf6.out.dis.dis_indices(
        ia, ja, ncells, nlayer, nrow, ncol
    )
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


def test_dis_indices__single_column():
    # Note: ia and ja use 1-based indexing as produced by modflow6!
    # Cell numbering, top-view:
    #
    # +---+
    # | 1 |
    # +---+
    # | 2 |
    # +---+
    # | 3 |
    # +---+
    # | 4 |
    # +---+
    ia = np.array([1, 2, 4, 6, 7])
    ja = np.array([2, 1, 3, 2, 4, 3])
    # nja index    0  1  2  3  4  5
    ncells = 4
    nlayer = 1
    nrow = 4
    ncol = 1
    right, front, lower = imod.mf6.out.dis.dis_indices(
        ia, ja, ncells, nlayer, nrow, ncol
    )
    assert right.shape == front.shape == lower.shape == (nlayer, nrow, ncol)
    assert (lower == -1).all()  # No lower connections
    assert (right == -1).all()  # No right connections
    front_expected = np.array(
        [
            [
                0,
            ],
            [
                2,
            ],
            [
                4,
            ],
            [
                -1,
            ],
        ]
    )
    assert np.allclose(front[0], front_expected)


def test_dis_indices__single_row_column():
    # Note: ia and ja use 1-based indexing as produced by modflow6!
    # Cell numbering, vertical-view:
    #
    # +---+
    # | 1 |
    # +---+
    # | 2 |
    # +---+
    # | 3 |
    # +---+
    # | 4 |
    # +---+
    ia = np.array([1, 2, 4, 6, 7])
    ja = np.array([2, 1, 3, 2, 4, 3])
    # nja index    0  1  2  3  4  5
    ncells = 4
    nlayer = 4
    nrow = 1
    ncol = 1
    right, front, lower = imod.mf6.out.dis.dis_indices(
        ia, ja, ncells, nlayer, nrow, ncol
    )
    assert right.shape == front.shape == lower.shape == (nlayer, nrow, ncol)
    assert (right == -1).all()  # No right connections
    assert (front == -1).all()  # No front connections
    lower_expected = np.array(
        [
            [
                0,
            ],
            [
                2,
            ],
            [
                4,
            ],
            [
                -1,
            ],
        ]
    )
    assert np.allclose(lower[:, 0], lower_expected)


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
    right, front, lower = imod.mf6.out.dis.dis_indices(
        ia, ja, ncells, nlayer, nrow, ncol
    )
    assert (right == -1).all()  # No right connection
    front_expected = np.array(
        [
            [0, 3, -1],
            [-1, -1, -1],
            [9, 12, -1],
        ]
    )
    assert np.allclose(front.reshape(nlayer, nrow), front_expected)
    lower_expected = np.array(
        [
            [1, 4, -1],
            [7, 4, -1],
            [-1, -1, -1],
        ]
    )
    assert np.allclose(lower.reshape(nlayer, nrow), lower_expected)


def test_mf6_csr_to_coo():
    indptr = np.array([1, 1, 2, 4, 6])
    indices = np.array([1, 2, 3, 4, 5])
    i, j = imod.mf6.out.disv.mf6_csr_to_coo(indptr, indices)
    np.testing.assert_array_equal(i, [1, 2, 2, 3, 3])
    np.testing.assert_array_equal(j, indices - 1)


def test_disv_indices():
    r"""
    Top view of grid:
          ______
        /|      |\
       / |      | \
      /  |      |  \
     / 1 |  2   | 3 \
    /____|______|____\

    Vertical cross-section of idomain:

    ┌─────┌─────┌─────┐
    │     │     │     │
    │  0  │  1  │  0  │
    │    1│    2│    3│
    ┌─────┌─────┌─────┐
    │     │     │     │
    │  0  │ -1  │  1  │
    │    4│    5│    6│
    ┌─────┌─────┌─────┐
    │     │     │     │
    │  1  │  1  │  1  │
    │    7│    8│    9│
    └─────└─────└─────┘ 

    nlayer = 3, nface = 3
    1-based cell number in lower right corner.
    """
    nlayer = 3
    nface = 3
    shape = (nlayer, nface)
    idomain = np.zeros(shape, dtype=int)
    idomain[0, 1] = 1
    idomain[1, 1] = -1
    idomain[1, 2] = 1
    idomain[2, :] = 1

    # i = 2, 6, 7, 8
    # j = 8, 9, 8, 9
    ia = np.array([0, 0, 1, 1, 1, 1, 2, 3, 4])
    ja = np.array([8, 9, 8, 9])

    # Top view: 3 cells
    edge_face_connectivity = np.array(
        [
            [0, -1],
            [0, -1],
            [0, 1],  # 2
            [1, -1],
            [1, -1],
            [1, 2],  # 5
            [2, -1],
            [2, -1],
        ]
    )
    n_edge = len(edge_face_connectivity)
    expected_lower = np.full(shape, -1)
    expected_lower[0, 1] = 0
    expected_lower[1, 1] = 0
    expected_lower[1, 2] = 1
    expected_horizontal = np.full((nlayer, n_edge), -1)
    expected_horizontal[2, 2] = 2
    expected_horizontal[2, 5] = 3

    lower, horizontal = imod.mf6.out.disv.disv_indices(
        ia, ja, idomain, edge_face_connectivity
    )
    np.testing.assert_array_equal(lower, expected_lower)
    np.testing.assert_array_equal(horizontal, expected_horizontal)


def test_open_cbc__dis(twri_result):
    modeldir = twri_result
    with imod.util.cd(modeldir):
        cbc = imod.mf6.open_cbc("GWF_1/GWF_1.cbc", "GWF_1/dis.dis.grb")
        assert isinstance(cbc, dict)
        # packagename_packagetype
        assert sorted(cbc.keys()) == [
            "chd_chd",
            "drn_drn",
            "flow-front-face",
            "flow-lower-face",
            "flow-right-face",
            "wel_wel",
        ]
        for array in cbc.values():
            assert array.shape == (1, 3, 15, 15)
            assert isinstance(array, xr.DataArray)
            assert isinstance(array.data, dask.array.Array)

            # Test if no errors are thrown if the array is loaded into memory
            array.load()


def test_open_cbc__dis_multi_drn_1_cell(twri_result_9_drn_in_1_cell):
    """
    Test if the cbc file is read correctly when there are multiple drain cells
    in one cell. Values should be summed.
    """
    modeldir = twri_result_9_drn_in_1_cell
    with imod.util.cd(modeldir):
        cbc = imod.mf6.open_cbc("GWF_1/GWF_1.cbc", "GWF_1/dis.dis.grb")
        assert isinstance(cbc, dict)
        assert sorted(cbc.keys()) == [
            "chd_chd",
            "drn_drn",
            "flow-front-face",
            "flow-lower-face",
            "flow-right-face",
            "wel_wel",
        ]
        drn = cbc["drn_drn"]
        assert drn.shape == (1, 3, 15, 15)
        assert isinstance(drn, xr.DataArray)
        assert isinstance(drn.data, dask.array.Array)

        # Test if no errors are thrown if the array is loaded into memory
        drn.load()

        actual_value = drn.sel(time=1.0, layer=1)[6, 9]
        np.testing.assert_approx_equal(actual_value, -24.7, significant=1)


def test_open_cbc__dis_transient(transient_twri_result):
    modeldir = transient_twri_result
    with imod.util.cd(modeldir):
        cbc = imod.mf6.open_cbc("GWF_1/GWF_1.cbc", "GWF_1/dis.dis.grb")
        assert isinstance(cbc, dict)
        assert sorted(cbc.keys()) == [
            "chd_chd",
            "drn_drn",
            "flow-front-face",
            "flow-lower-face",
            "flow-right-face",
            "sto-ss",
            "wel_wel",
        ]
        for array in cbc.values():
            assert array.shape == (30, 3, 15, 15)
            assert isinstance(array, xr.DataArray)
            assert isinstance(array.data, dask.array.Array)

            # Test if no errors are thrown if the array is loaded into memory
            array.load()


def test_open_cbc__dis_datetime(transient_twri_result):
    modeldir = transient_twri_result
    with imod.util.cd(modeldir):
        cbc = imod.mf6.open_cbc(
            "GWF_1/GWF_1.cbc",
            "GWF_1/dis.dis.grb",
            simulation_start_time="01-01-1999",
            time_unit="d",
        )

    for array in cbc.values():
        assert array.coords["time"].dtype == np.dtype("datetime64[ns]")


def test_open_cbc__dis_transient_unconfined(transient_unconfined_twri_result):
    modeldir = transient_unconfined_twri_result
    with imod.util.cd(modeldir):
        cbc = imod.mf6.open_cbc("GWF_1/GWF_1.cbc", "GWF_1/dis.dis.grb")
        assert isinstance(cbc, dict)
        assert sorted(cbc.keys()) == [
            "chd_chd",
            "drn_drn",
            "flow-front-face",
            "flow-lower-face",
            "flow-right-face",
            "npf-qx",
            "npf-qy",
            "npf-qz",
            "npf-sat",
            "sto-ss",
            "sto-sy",
            "wel_wel",
        ]
        for array in cbc.values():
            assert array.shape == (30, 3, 15, 15)
            assert isinstance(array, xr.DataArray)
            assert isinstance(array.data, dask.array.Array)

            # Test if no errors are thrown if the array is loaded into memory
            array.load()


def test_open_cbc__disv(circle_result):
    modeldir = circle_result
    with imod.util.cd(modeldir):
        cbc = imod.mf6.open_cbc("GWF_1/GWF_1.cbc", "GWF_1/disv.disv.grb")
        assert isinstance(cbc, dict)
        assert sorted(cbc.keys()) == [
            "chd_chd",
            "flow-horizontal-face",
            "flow-horizontal-face-x",
            "flow-horizontal-face-y",
            "flow-lower-face",
        ]
        for key, array in cbc.items():
            if key in ("chd_chd", "flow-lower-face"):
                assert array.shape == (52, 2, 216)
                assert array.dims[-1] == array.ugrid.grid.face_dimension
            else:
                assert array.shape == (52, 2, 342)
                assert array.dims[-1] == array.ugrid.grid.edge_dimension
            assert isinstance(array, xu.UgridDataArray)
            assert isinstance(array.data, dask.array.Array)

            # Test if no errors are thrown if the array is loaded into memory
            array.load()


def test_open_cbc__disv_offset(circle_result__offset_origins):
    modeldir = circle_result__offset_origins
    with imod.util.cd(modeldir):
        cbc = imod.mf6.open_cbc("GWF_1/GWF_1.cbc", "GWF_1/disv.disv.grb")
        assert isinstance(cbc, dict)
        assert sorted(cbc.keys()) == [
            "chd_chd",
            "flow-horizontal-face",
            "flow-horizontal-face-x",
            "flow-horizontal-face-y",
            "flow-lower-face",
        ]
        for key, array in cbc.items():
            if key in ("chd_chd", "flow-lower-face"):
                assert array.shape == (52, 2, 216)
                assert array.dims[-1] == array.ugrid.grid.face_dimension
            else:
                assert array.shape == (52, 2, 342)
                assert array.dims[-1] == array.ugrid.grid.edge_dimension
            assert isinstance(array, xu.UgridDataArray)
            assert isinstance(array.data, dask.array.Array)

            # Test if no errors are thrown if the array is loaded into memory
            array.load()
            # Test if offset of 10_000 is applied correctly
            xy_vertices = array.ugrid.grid.get_coordinates("mesh2d_nNodes")
            xy_faces = array.ugrid.grid.get_coordinates("mesh2d_nFaces")
            assert np.max(xy_vertices) <= 11_000
            assert np.min(xy_vertices) >= 9_000
            assert np.max(xy_faces) <= 11_000
            assert np.min(xy_faces) >= 9_000


def test_open_cbc__disv_datetime(circle_result):
    modeldir = circle_result
    with imod.util.cd(modeldir):
        cbc = imod.mf6.open_cbc(
            "GWF_1/GWF_1.cbc",
            "GWF_1/disv.disv.grb",
            simulation_start_time="01-01-1999",
            time_unit="d",
        )

    for array in cbc.values():
        assert array.coords["time"].dtype == np.dtype("datetime64[ns]")


def test_open_cbc__disv_sto(circle_result_sto):
    """With saved storage fluxes, which are saved as METH1"""
    modeldir = circle_result_sto
    with imod.util.cd(modeldir):
        cbc = imod.mf6.open_cbc("GWF_1/GWF_1.cbc", "GWF_1/disv.disv.grb")
        assert isinstance(cbc, dict)
        assert sorted(cbc.keys()) == [
            "chd_chd",
            "flow-horizontal-face",
            "flow-horizontal-face-x",
            "flow-horizontal-face-y",
            "flow-lower-face",
            "sto-ss",
        ]
        for key, array in cbc.items():
            if key in ("chd_chd", "flow-lower-face", "sto-ss"):
                assert array.shape == (52, 2, 216)
                assert array.dims[-1] == array.ugrid.grid.face_dimension
            else:
                assert array.shape == (52, 2, 342)
                assert array.dims[-1] == array.ugrid.grid.edge_dimension
            assert isinstance(array, xu.UgridDataArray)
            assert isinstance(array.data, dask.array.Array)

            # Test if no errors are thrown if the array is loaded into memory
            array.load()
