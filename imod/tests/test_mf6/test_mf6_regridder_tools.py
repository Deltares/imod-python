import copy

import pytest

from imod.mf6.regridding_tools import RegridderInstancesCollection


def test_instance_collection_returns_same_instance_when_name_and_method_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    collection = RegridderInstancesCollection()

    new_grid = copy.deepcopy(grid)

    first_instance = collection.get_regridder(
        "OverlapRegridder", grid, new_grid, "harmonic_mean"
    )
    second_instance = collection.get_regridder(
        "OverlapRegridder", grid, new_grid, "harmonic_mean"
    )

    assert first_instance == second_instance


def test_instance_collection_returns_different_instance_when_name_does_not_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    collection = RegridderInstancesCollection()

    new_grid = copy.deepcopy(grid)

    first_instance = collection.get_regridder(
        "CentroidLocatorRegridder", source=grid, target=new_grid
    )
    second_instance = collection.get_regridder(
        "BarycentricInterpolator", source=grid, target=new_grid
    )

    assert first_instance != second_instance


def test_instance_collection_returns_different_instance_when_method_does_not_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    collection = RegridderInstancesCollection()

    new_grid = copy.deepcopy(grid)

    first_instance = collection.get_regridder(
        "OverlapRegridder", source=grid, target=new_grid, method="geometric_mean"
    )
    second_instance = collection.get_regridder(
        "OverlapRegridder", source=grid, target=new_grid, method="harmonic_mean"
    )

    assert first_instance != second_instance


def test_error_messages(basic_unstructured_dis):
    grid, _, _ = basic_unstructured_dis
    collection = RegridderInstancesCollection()

    new_grid = copy.deepcopy(grid)
    with pytest.raises(
        ValueError, match="BarycentricInterpolator does not support methods"
    ):
        _ = collection.get_regridder(
            "BarycentricInterpolator",
            source=grid,
            target=new_grid,
            method="geometric_mean",
        )

    with pytest.raises(
        ValueError, match="unknown regridder type Non-existing regridder"
    ):
        _ = collection.get_regridder(
            "Non-existing regridder",
            source=grid,
            target=new_grid,
            method="geometric_mean",
        )
