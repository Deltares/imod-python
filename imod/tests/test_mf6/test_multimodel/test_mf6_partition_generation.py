import numpy as np
import xarray as xr
import xugrid as xu

from imod.mf6 import Modflow6Simulation


def test_partition_2d_unstructured(circle_model: Modflow6Simulation):
    for nr_partitions in range(1, 20):
        label_array = circle_model.create_partition_labels(nr_partitions)
        assert isinstance(label_array, xu.UgridDataArray)
        # check that the labes up to nr_partitions -1 appear in the label array, and not any others
        unique, counts = np.unique(label_array, return_counts=True)
        assert np.all(
            counts > 4
        )  # A partition should have at least 4 triangles to have  a cell that is not an exchange cell
        assert len(unique) == nr_partitions
        assert max(unique) == nr_partitions - 1
        assert min(unique) == 0
        assert np.issubdtype(label_array.dtype, np.integer)
        assert label_array.name == "label"


def test_partition_2d_unstructured_with_weights(circle_model: Modflow6Simulation):
    weights = circle_model["GWF_1"].domain.isel(layer=0).copy()
    weights[:50] = 10
    for nr_partitions in range(2, 20):
        label_array = circle_model.create_partition_labels(nr_partitions, weights)
        assert isinstance(label_array, xu.UgridDataArray)
        # check that the labes up to nr_partitions -1 appear in the label array, and not any others
        unique, counts = np.unique(label_array, return_counts=True)
        assert len(unique) == nr_partitions
        assert max(unique) == nr_partitions - 1
        assert min(unique) == 0
        assert (
            np.unique(counts).size != 1
        )  # Partitions should have different number of cells
        assert np.issubdtype(label_array.dtype, np.integer)
        assert label_array.name == "label"


def test_partition_2d_structured(twri_model: Modflow6Simulation):
    # We skip a few partition numbers which would give an error. This would
    # happen if the number of partitions in the x or y direction would result in
    # less then 3 gridblocks per partition. For example if we ask for 7
    # partitions it would result in splitting the 15x15 domain in 7 partitions
    # on 1 axis and 1 on the other axis. That would give 15/7 = 2 gridblocks per
    # partition along one of the axis However asking for 8 partitions would be
    # split as 2 and 4 partitions on x and y and would work.
    partition_numbers = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20]
    for nr_partitions in partition_numbers:
        label_array = twri_model.create_partition_labels(nr_partitions)
        assert isinstance(label_array, xr.DataArray)
        unique, counts = np.unique(label_array, return_counts=True)
        # Check that the labes up to nr_partitions -1 appear in the label array, and not any others
        assert np.all(
            counts >= 9
        )  # A partition should be at least 3x3 to have a cell that is not an exchange cell
        assert len(unique) == nr_partitions
        assert max(unique) == nr_partitions - 1
        assert min(unique) == 0
        assert np.issubdtype(label_array.dtype, np.integer)
        assert label_array.name == "label"


def test_partition_2d_structured_with_weights(twri_model: Modflow6Simulation):
    # We skip a few partition numbers which would give an error. This would
    # happen if the number of partitions in the x or y direction would result in
    # less then 3 gridblocks per partition. For example if we ask for 7
    # partitions it would result in splitting the 15x15 domain in 7 partitions
    # on 1 axis and 1 on the other axis. That would give 15/7 = 2 gridblocks per
    # partition along one of the axis However asking for 8 partitions would be
    # split as 2 and 4 partitions on x and y and would work.
    weights = twri_model["GWF_1"].domain.isel(layer=0).copy()
    weights[:5, :5] = 10
    partition_numbers = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20]
    for nr_partitions in partition_numbers:
        label_array = twri_model.create_partition_labels(nr_partitions, weights)
        assert isinstance(label_array, xr.DataArray)
        unique, counts = np.unique(label_array, return_counts=True)
        # Check that the labes up to nr_partitions -1 appear in the label array, and not any others
        assert len(unique) == nr_partitions
        assert max(unique) == nr_partitions - 1
        assert min(unique) == 0
        assert (
            np.unique(counts).size != 1
        )  # Partitions should have different number of cells
        assert np.issubdtype(label_array.dtype, np.integer)
        assert label_array.name == "label"
