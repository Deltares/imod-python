from pathlib import Path

import numpy as np
import pytest

from imod.mf6.partitioned_simulation_postprocessing import (
    get_grb_file_path,
    merge_heads,
)
from imod.mf6.simulation import Modflow6Simulation, get_models


def test_find_grb_file(tmp_path: Path):
    # Arrange
    grb_path = tmp_path / "modelname.grb"
    with open(grb_path, "a") as file:
        file.write("grb file content")

    # Act
    grb_file = get_grb_file_path(tmp_path)

    # Assert
    assert grb_file.name == "modelname.grb"


def test_find_no_grb_file(tmp_path: Path):
    # Act, Assert
    with pytest.raises(RuntimeError):
        _ = get_grb_file_path(tmp_path)


def test_find_multiple_grb_files(tmp_path: Path):
    # Arrange
    grb_path1 = tmp_path / "modelname1.grb"
    with open(grb_path1, "a") as file:
        file.write("grb file content")

    grb_path2 = tmp_path / "modelname2.grb"
    with open(grb_path2, "a") as file:
        file.write("grb file content")

    # Act, Assert
    with pytest.raises(RuntimeError):
        _ = get_grb_file_path(tmp_path)


@pytest.mark.usefixtures("split_transient_twri_model")
def test_import_heads_structured(
    tmp_path: Path, split_transient_twri_model: Modflow6Simulation
):
    # Arrange
    split_simulation = split_transient_twri_model
    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    submodel_names = list(get_models(split_simulation).keys())

    # Act
    merged_heads = merge_heads(tmp_path, submodel_names)

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


@pytest.mark.usefixtures("circle_partitioned")
def test_import_heads_unstructured(tmp_path, circle_partitioned):
    # Arrange

    circle_partitioned.write(tmp_path, binary=False)
    circle_partitioned.run()

    submodel_names = list(get_models(circle_partitioned).keys())

    # Act
    merged_heads = merge_heads(tmp_path, submodel_names)

    # Assert
    assert np.allclose(merged_heads.coords["layer"].values, [1, 2])
    assert np.allclose(merged_heads.coords["time"].values, [1.0])
    assert np.allclose(merged_heads.coords["mesh2d_nFaces"].values, list(range(216)))
