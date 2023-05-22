import copy

import pytest

from imod.mf6.regridding_tools import RegridderInstancesCollection


def test_instance_collection_returns_same_instance_when_name_and_method_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderInstancesCollection(grid, new_grid)

    first_instance = collection.get_regridder("OverlapRegridder", "harmonic_mean")
    second_instance = collection.get_regridder("OverlapRegridder", "harmonic_mean")

    assert first_instance == second_instance


def test_instance_collection_returns_different_instance_when_name_does_not_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderInstancesCollection(grid, new_grid)

    first_instance = collection.get_regridder("CentroidLocatorRegridder")
    second_instance = collection.get_regridder("BarycentricInterpolator")

    assert first_instance != second_instance


def test_instance_collection_returns_different_instance_when_method_does_not_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderInstancesCollection(grid, new_grid)

    first_instance = collection.get_regridder("OverlapRegridder", "geometric_mean")
    second_instance = collection.get_regridder("OverlapRegridder", "harmonic_mean")

    assert first_instance != second_instance


def test_error_messages(basic_unstructured_dis):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderInstancesCollection(grid, new_grid)
    with pytest.raises(
        ValueError, match="BarycentricInterpolator does not support methods"
    ):
        _ = collection.get_regridder(
            "BarycentricInterpolator",
            method="geometric_mean",
        )

    with pytest.raises(
        ValueError, match="unknown regridder type Non-existing regridder"
    ):
        _ = collection.get_regridder(
            "Non-existing regridder",
            method="geometric_mean",
        )

    with pytest.raises(ValueError, match="Invalid regridding method"):
        _ = collection.get_regridder(
            "RelativeOverlapRegridder",
            method="non-existing function",
        )
