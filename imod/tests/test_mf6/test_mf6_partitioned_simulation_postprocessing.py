from pathlib import Path

import pytest

from imod.mf6.partitioned_simulation_postprocessing import (
    get_grb_file_path,
    merge_heads,
)
from imod.mf6.simulation import Modflow6Simulation, get_models
import imod
import numpy as np
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


@pytest.mark.usefixtures("setup_split_simulation")
def test_import_heads_structured(
    tmp_path: Path, setup_split_simulation: Modflow6Simulation
):
    # Arrange
    split_simulation = setup_split_simulation
    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    submodel_names = list(get_models(split_simulation).keys())

    # Act
    merged_heads = merge_heads(tmp_path, submodel_names)

    #assert
    assert np.allclose(merged_heads.coords["x"].values , [ 2500.,  7500., 12500., 17500., 22500., 27500., 32500., 37500.,
       42500., 47500., 52500., 57500., 62500., 67500., 72500.])
    assert np.allclose(merged_heads.coords["y"].values , [ 2500.,  7500., 12500., 17500., 22500., 27500., 32500., 37500.,
       42500., 47500., 52500., 57500., 62500., 67500., 72500.])
    assert np.allclose(merged_heads.coords["layer"].values ,[1, 2, 3])
    assert np.allclose(merged_heads.coords["time"].values ,[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
       14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
       27., 28., 29., 30.])
    
@pytest.mark.usefixtures("make_circle_partitioned")
def test_import_heads_unstructured(tmp_path, make_circle_partitioned):
    # Arrange
    split_simulation = make_circle_partitioned
    split_simulation.write(tmp_path, binary=False)
    split_simulation.run()

    submodel_names = list(get_models(split_simulation).keys())

    # Act
    merged_heads = merge_heads(tmp_path, submodel_names)

    #Assert
    assert np.allclose(merged_heads.coords["x"].values , [ 2500.,  7500., 12500., 17500., 22500., 27500., 32500., 37500.,
       42500., 47500., 52500., 57500., 62500., 67500., 72500.])
    assert np.allclose(merged_heads.coords["y"].values , [ 2500.,  7500., 12500., 17500., 22500., 27500., 32500., 37500.,
       42500., 47500., 52500., 57500., 62500., 67500., 72500.])
    assert np.allclose(merged_heads.coords["layer"].values ,[1, 2, 3])
    assert np.allclose(merged_heads.coords["time"].values ,[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
       14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
       27., 28., 29., 30.])
