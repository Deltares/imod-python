import copy
from typing import Optional

import xarray as xr
import xugrid as xu
from plum import Dispatcher

from imod.typing import GridDataArray
from imod.typing.grid import as_ugrid_dataarray

# create dispatcher instance to limit scope of typedispatching
dispatch = Dispatcher()


def _validate_weights(idomain: GridDataArray, weights: GridDataArray) -> None:
    idomain_top = copy.deepcopy(idomain.isel(layer=0))
    if not isinstance(weights, type(idomain)):
        raise TypeError(
            f"weights should be of type {type(idomain)}, but is of type {type(weights)}"
        )
    if idomain_top.sizes != weights.sizes:
        raise ValueError(
            f"Weights do not have the appropriate size. Expected {idomain_top.sizes}, got {weights.sizes}"
        )


@dispatch
def _partition_idomain(weights: xu.UgridDataArray, npartitions: int) -> GridDataArray:
    """
    Create a label array for unstructured grids using PyMetis.
    """
    labels = weights.ugrid.label_partitions(n_part=npartitions)
    labels = labels.rename("label")
    return labels


@dispatch  # type: ignore[no-redef]
def _partition_idomain(weights: xr.DataArray, npartitions: int) -> GridDataArray:
    """
    Convert to UgridDataArray to use xugrid to call PyMetis to create a label
    array for structured grids. Call as_ugrid_dataarray to used cached results
    of ``Ugrid2d.from_structured`` if available, to save some costly
    conversions.
    """
    weights_uda = as_ugrid_dataarray(weights)
    labels_uda = weights_uda.ugrid.label_partitions(n_part=npartitions)
    dim = labels_uda.ugrid.grid.core_dimension
    labels = labels_uda.ugrid.rasterize_like(weights).astype(int)
    labels = labels.drop_vars(dim)
    labels = labels.rename("label")
    return labels


def create_partition_labels(
    idomain: GridDataArray,
    npartitions: int,
    weights: Optional[GridDataArray] = None,
) -> GridDataArray:
    """
    Returns a label array: a 2d array with a similar size to the top layer of
    idomain. Every array element is the partition number to which the column of
    gridblocks of idomain at that location belong. This is provided to
    :meth:`imod.mf6.Modflow6Simulation.split` to partition the model.

    Parameters
    ----------
    idomain : GridDataArray
        idomain-like integer array. >0 sets cells to active, 0 sets cells to inactive,
        <0 sets cells to vertical passthrough.
    npartitions : int
        The number of partitions to create.
    weights : xarray.DataArray, xugrid.UgridDataArray, optional
        The weights to use for partitioning. The weights should be a 2d array
        with the same size as the top layer of idomain. The weights are used to
        determine the size of each partition. The weights should be positive
        integers. If not provided, active cells (idomain > 0) are summed across
        layers and passed on as weights. If None, the idomain is used to compute
        weights.

    Returns
    -------
    xr.DataArray or xugrid.UgridDataArray
        A label array with the same size as the top layer of idomain, where each
        element is the partition number to which the column of gridblocks at that
        location belongs.

    Examples
    --------
    Create partition labels for a simulation:

    >>> partition_labels = create_partition_labels(idomain, npartitions=4)

    You can provide these labels to the :meth:`imod.mf6.Modflow6Simulation.split` method

    >>> mf6_splitted = mf6_sim.split(label_array)

    You can also provide weights to the partitioning, which will influence the
    size of each partition. For example, if you want to create a uniform
    partitioning, you can use:

    >>> weights = xr.ones_like(idomain)
    >>> label_array = create_partition_labels(idomain, n_partitions=4, weights=weights)
    """
    if npartitions <= 0:
        raise ValueError("You should create at least 1 partition")

    if weights is None:
        weights = (idomain > 0).sum(dim="layer").astype(int)
    else:
        _validate_weights(idomain, weights)

    return _partition_idomain(weights, npartitions)
