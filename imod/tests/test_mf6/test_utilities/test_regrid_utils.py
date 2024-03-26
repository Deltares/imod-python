import copy
import pickle

import pytest
from xugrid import OverlapRegridder

from imod.mf6 import Dispersion
from imod.mf6.utilities.regrid import RegridderType, RegridderWeightsCache


def test_regridders_weight_cache_returns_similar_instance_when_enum_and_method_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderWeightsCache(grid, new_grid)

    first_instance = collection.get_regridder(grid, new_grid, RegridderType.OVERLAP, "harmonic_mean")
    second_instance = collection.get_regridder(grid, new_grid, RegridderType.OVERLAP, "harmonic_mean")

    assert hash (pickle.dumps(first_instance.weights["__regrid_data"])) == hash (pickle.dumps(second_instance.weights["__regrid_data"]))


def test_regridders_weight_cache_combining_different_instantiation_parmeters(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderWeightsCache(grid, new_grid)

    first_instance = collection.get_regridder(grid, new_grid, RegridderType.OVERLAP, "harmonic_mean")
    second_instance = collection.get_regridder(grid, new_grid, OverlapRegridder, "harmonic_mean")

    assert hash (pickle.dumps(first_instance.weights["__regrid_data"])) == hash (pickle.dumps(second_instance.weights["__regrid_data"]))

def test_regridders_weight_cache_returns_different_instance_when_name_does_not_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderWeightsCache(grid, new_grid)

    first_instance = collection.get_regridder(grid, new_grid, RegridderType.CENTROIDLOCATOR)
    second_instance = collection.get_regridder(grid, new_grid, RegridderType.BARYCENTRIC)

    assert hash (pickle.dumps(first_instance)) != hash (pickle.dumps(second_instance))

def test_regridders_weight_cache_returns_different_instance_when_method_does_not_match(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderWeightsCache(grid, new_grid)

    first_instance = collection.get_regridder(grid, new_grid, RegridderType.OVERLAP, "geometric_mean")
    second_instance = collection.get_regridder(grid, new_grid, RegridderType.OVERLAP, "harmonic_mean")

    assert hash (pickle.dumps(first_instance)) != hash (pickle.dumps(second_instance))

def test_non_regridder_cannot_be_instantiated(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderWeightsCache(grid, new_grid)

    # we create a class with the same constructor-signature as a regridder has, but it is not a regridder
    # still, it is an abc.ABCMeta
    class nonregridder(Dispersion):
        def __init__(self, sourcegrid, targetgrid, method):
            pass

    with pytest.raises(ValueError):
        _ = collection.get_regridder(grid, new_grid, nonregridder, "geometric_mean")


def test_regridders_weight_cache_grows_up_to_size_limit(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)
    cache_size = 3
    collection = RegridderWeightsCache(grid, new_grid, cache_size)

    _ = collection.get_regridder(grid, new_grid, RegridderType.OVERLAP, "harmonic_mean")
    _ = collection.get_regridder(grid, new_grid, RegridderType.BARYCENTRIC)
    _ =  collection.get_regridder(grid, new_grid, RegridderType.CENTROIDLOCATOR)
    assert len(collection.weights_cache) ==cache_size
    _ =  collection.get_regridder(grid, new_grid, RegridderType.RELATIVEOVERLAP, method="conductance") 
    assert len(collection.weights_cache) == cache_size 
  


def test_error_messages(basic_unstructured_dis):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderWeightsCache(grid, new_grid)
    with pytest.raises(TypeError):
        _ = collection.get_regridder(grid, new_grid, 
            RegridderType.BARYCENTRIC,
            method="geometric_mean",
        )

    with pytest.raises(ValueError):
        _ = collection.get_regridder(grid, new_grid, 
            RegridderType.OVERLAP,
            method="non-existing function",
        )


def test_create_regridder_from_class_not_enum(
    basic_unstructured_dis,
):
    grid, _, _ = basic_unstructured_dis
    new_grid = copy.deepcopy(grid)

    collection = RegridderWeightsCache(grid, new_grid)
    regridder = collection.get_regridder(grid, new_grid, OverlapRegridder, "harmonic_mean")

    assert isinstance(regridder, OverlapRegridder)
