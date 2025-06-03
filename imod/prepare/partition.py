import copy
from typing import Optional

import xarray as xr
import xugrid as xu
from plum import Dispatcher

from imod.common.interfaces.isimulation import ISimulation
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
    simulation: ISimulation,
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
    simulation : Modflow6Simulation
        The simulation to partition. It must have exactly one flow model.
    npartitions : int
        The number of partitions to create.
    weights : xarray.DataArray, xugrid.UgridDataArray, optional
        The weights to use for partitioning. The weights should be a 2d array
        with the same size as the top layer of idomain. The weights are used to
        determine the size of each partition. The weights should be positive
        integers. If not provided, active cells (idomain > 0) are summed across
        layers and passed on as weights.
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

    if weights is None:
        weights = (idomain > 0).sum(dim="layer").astype(int)
    else:
        _validate_weights(idomain, weights)

    return _partition_idomain(weights, npartitions)
