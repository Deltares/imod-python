import copy
from math import sqrt
from typing import List, Tuple

import xarray as xr
import xugrid as xu
from plum import Dispatcher

from imod.mf6.simulation import Modflow6Simulation
from imod.typing import GridDataArray

# create dispatcher instance to limit scope of typedispatching
dispatch = Dispatcher()


def get_label_array(simulation: Modflow6Simulation, npartitions: int) -> GridDataArray:
    """
    Returns a label array: a 2d array with a similar size to the top layer of
    idomain. Every array element is the partition number to which the column of
    gridblocks of idomain at that location belong.
    """
    gwf_models = simulation.get_models_of_type("gwf6")
    if len(gwf_models) != 1:
        raise ValueError(
            "for partitioning a simulation to work, it must have exactly 1 flow model"
        )
    if npartitions <= 0:
        raise ValueError("You should create at least 1 partition")

    flowmodel = list(gwf_models.values())[0]
    idomain = flowmodel.domain
    idomain_top = copy.deepcopy(idomain.isel(layer=0))

    return _partition_idomain(idomain_top, npartitions)


@dispatch
def _partition_idomain(
    idomain_grid: xu.UgridDataArray, npartitions: int
) -> GridDataArray:
    """
    Create a label array for unstructured grids using xugrid and, though it, Metis.
    """
    labels = idomain_grid.ugrid.grid.label_partitions(n_part=npartitions)
    labels = labels.rename("idomain")
    return labels


@dispatch  # type: ignore[no-redef]
def _partition_idomain(idomain_grid: xr.DataArray, npartitions: int) -> GridDataArray:
    """
    Create a label array for structured grids by creating rectangular
    partitions. It factors the requested number of partitions into the two
    factors closest to the square root. The axis with the most gridbblocks will
    be split into the largest number of partitions.
    """

    # get axis sizes
    x_axis_size = idomain_grid.shape[0]
    y_axis_size = idomain_grid.shape[1]

    smallest_factor, largest_factor = _mid_size_factors(npartitions)
    if x_axis_size < y_axis_size:
        nr_partitions_x, nr_partitions_y = smallest_factor, largest_factor
    else:
        nr_partitions_y, nr_partitions_x = smallest_factor, largest_factor

    x_partition = _partition_1d(nr_partitions_x, x_axis_size)
    y_partition = _partition_1d(nr_partitions_y, y_axis_size)

    ipartition = -1
    for ipartx in x_partition:
        start_x, stop_x = ipartx
        for iparty in y_partition:
            start_y, stop_y = iparty
            ipartition = ipartition + 1
            idomain_grid.values[start_x : stop_x + 1, start_y : stop_y + 1] = ipartition
    return idomain_grid


def _partition_1d(nr_partitions: int, axis_size: int) -> List[Tuple]:
    """
    Returns tuples with start and stop positions of partitions when partitioning
    an axis of length nr_indices into nr_partitions. Partitions need to be at
    least 3 gridblocks in size. If this cannot be done, it throws an error. When
    the number of gridblocks on the axis is not divisible by the number of
    partitions, then any leftover cells are added in the last partition. For
    example if we partition an axis of 25 cells into 3 partitions, then the
    number of cells per partition will be 8 for the first 2 partitions,but the
    last partition will contain 9 cells.
    """

    # validate input
    if nr_partitions <= 0:
        raise ValueError(
            "error while partitioning an axis. Create at least 1 partition"
        )
    if axis_size <= 0:
        raise ValueError(
            "error while partitioning an axis. Axis sized should be positive"
        )

    # quick exit if 1 partition is requested
    if nr_partitions == 1:
        return [(0, axis_size - 1)]

    # compute partition size. round fractions down.
    cells_per_partition = int(axis_size / nr_partitions)
    if cells_per_partition < 3:
        raise ValueError(
            "error while partitioning an axis. The number of partitions is too large, We should have at least 3 gridblocks in a partition along any axis."
        )

    # fill the partitions up to the penultimate.
    partitions = [
        (i * cells_per_partition, i * cells_per_partition + cells_per_partition - 1)
        for i in range(nr_partitions - 1)
    ]

    # now set the lat partition up to the end of the axis
    final = partitions[-1][1]
    partitions.append((final + 1, axis_size - 1))
    return partitions


def _mid_size_factors(number_partitions: int) -> Tuple[int, int]:
    """
    Returns the 2 factors of an integer that are closest to the square root
    (smallest first). Calling it on 27 would return 3 and 7; calling it on 25
    would return 5 fand 5, calling it on 13 would return 1 and 13.
    """

    factor = int(sqrt(number_partitions))
    while factor > 0:
        if number_partitions % factor == 0:
            break
        else:
            factor -= 1

    return factor, int(number_partitions / factor)
