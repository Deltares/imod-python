from pathlib import Path

import numpy as np

from imod.mf6.simulation import Modflow6Simulation


def test_import_heads_structured(
    tmp_path: Path, split_transient_twri_model: Modflow6Simulation
):
    # Arrange
    split_simulation = split_transient_twri_model
    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    # Act
    merged_heads = split_simulation.open_head()

    # Assert
    assert np.allclose(
        merged_heads.coords["x"].values,
        [
            2500.0,
            7500.0,
            12500.0,
            17500.0,
            22500.0,
            27500.0,
            32500.0,
            37500.0,
            42500.0,
            47500.0,
            52500.0,
            57500.0,
            62500.0,
            67500.0,
            72500.0,
        ],
    )
    assert np.allclose(
        merged_heads.coords["y"].values,
        [
            72500.0,
            67500.0,
            62500.0,
            57500.0,
            52500.0,
            47500.0,
            42500.0,
            37500.0,
            32500.0,
            27500.0,
            22500.0,
            17500.0,
            12500.0,
            7500.0,
            2500.0,
        ],
    )
    assert np.allclose(merged_heads.coords["layer"].values, [1, 2, 3])
    assert np.allclose(
        merged_heads.coords["time"].values,
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
            26.0,
            27.0,
            28.0,
            29.0,
            30.0,
        ],
    )


def test_import_heads_unstructured(tmp_path, circle_partitioned):
    # Arrange

    circle_partitioned.write(tmp_path, binary=False)
    circle_partitioned.run()

    # Act
    merged_heads = circle_partitioned.open_head()

    # Assert
    assert np.allclose(merged_heads.coords["layer"].values, [1, 2])
    assert np.allclose(
        merged_heads.coords["time"].values, list(np.arange(7.0, 365.0, 7.0))
    )
    assert np.allclose(merged_heads.coords["mesh2d_nFaces"].values, list(range(216)))


def test_import_balances_structured(
    tmp_path: Path, split_transient_twri_model: Modflow6Simulation
):
    # Arrange
    split_simulation = split_transient_twri_model
    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    # Act
    merged_balances = split_simulation.open_flow_budget()

    # Assert
    expected_keys = [
        "gwf-gwf",
        "chd_chd",
        "flow-right-face",
        "sto-ss",
        "flow-lower-face",
        "drn_drn",
        "flow-front-face",
    ]
    expected_coords = ["x", "y", "layer", "time", "dx", "dy"]
    expected_dims = ["time", "layer", "y", "x"]

    for key in expected_keys:
        assert key in merged_balances.keys()
        assert len(expected_coords) == len(merged_balances[key].coords)
        assert len(expected_dims) == len(merged_balances[key].dims)
        for coord in expected_coords:
            assert coord in merged_balances[key].coords
        for dim in expected_dims:
            assert dim in merged_balances[key].dims


def test_import_balances_unstructured(
    tmp_path: Path, circle_partitioned: Modflow6Simulation
):
    # Arrange
    split_simulation = circle_partitioned
    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    # Act
    merged_balances = split_simulation.open_flow_budget()

    # Assert
    expected_keys = [
        "chd_chd",
        "flow-horizontal-face",
        "gwf-gwf",
        "flow-horizontal-face-y",
        "flow-lower-face",
        "flow-horizontal-face-x",
    ]
    expected_dims_coords_faces = ["layer", "time", "mesh2d_nFaces"]
    expected_dims_coords_edges = ["layer", "time", "mesh2d_nEdges"]

    for key in expected_keys:
        expected_dims_coords = expected_dims_coords_faces
        if "flow-horizontal-face" in key:
            expected_dims_coords = expected_dims_coords_edges

        assert key in merged_balances.keys()
        assert len(expected_dims_coords) == len(merged_balances[key].coords)
        assert len(expected_dims_coords) == len(merged_balances[key].dims)

        for coord in expected_dims_coords:
            assert coord in merged_balances[key].coords
            assert coord in merged_balances[key].dims
