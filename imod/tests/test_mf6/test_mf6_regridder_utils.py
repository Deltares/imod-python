import copy

import pytest

from imod.mf6.regridding_utils import RegridderInstancesCollection
from imod.mf6.regridding_utils import RegridderType as rt


def test_instance_collection_returns_same_instance_when_name_and_method_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderInstancesCollection(grid, new_grid)

    first_instance = collection.get_regridder(rt.OVERLAP, "harmonic_mean")
    second_instance = collection.get_regridder(rt.OVERLAP, "harmonic_mean")

    assert first_instance == second_instance


def test_instance_collection_returns_different_instance_when_name_does_not_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderInstancesCollection(grid, new_grid)

    first_instance = collection.get_regridder(rt.CENTROIDLOCATOR)
    second_instance = collection.get_regridder(rt.BARYCENTRIC)

    assert first_instance != second_instance


def test_instance_collection_returns_different_instance_when_method_does_not_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderInstancesCollection(grid, new_grid)

    first_instance = collection.get_regridder(rt.OVERLAP, "geometric_mean")
    second_instance = collection.get_regridder(rt.OVERLAP, "harmonic_mean")

    assert first_instance != second_instance


def test_error_messages(basic_unstructured_dis):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderInstancesCollection(grid, new_grid)
    with pytest.raises(
        ValueError, match="failed to create a regridder <class 'xugrid.regrid.regridder.BarycentricInterpolator'> with method geometric_mean"
    ):
        _ = collection.get_regridder(
            rt.BARYCENTRIC,
            method="geometric_mean",
        )

    with pytest.raises(ValueError, match="failed to create a regridder <class 'xugrid.regrid.regridder.OverlapRegridder'> with method non-existing function"):
        _ = collection.get_regridder(
            rt.OVERLAP,
            method="non-existing function",
        )
