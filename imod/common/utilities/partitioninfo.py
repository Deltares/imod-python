from typing import List, NamedTuple, Optional

import numpy as np

from imod.typing import GridDataArray
from imod.typing.grid import ones_like


class PartitionInfo(NamedTuple):
    active_domain: GridDataArray
    id: int
    mpi_rank: int


def create_partition_info(
    submodel_labels: GridDataArray,
    submodel_label_to_mpi_rank: Optional[dict[int, int]] = None,
) -> List[PartitionInfo]:
    """
    A PartitionInfo is used to partition a model or package. The partition
    info's of a domain are created using a submodel_labels array. The
    submodel_labels provided as input should have the same shape as a single
    layer of the model grid (all layers are split the same way), and contains an
    integer value in each cell. Each cell in the model grid will end up in the
    submodel with the index specified by the corresponding label of that cell.
    The labels should be numbers between 0 and the number of partitions.
    """
    _validate_submodel_label_array(submodel_labels)

    unique_labels = np.unique(submodel_labels.values)

    partition_infos = []
    for label_id in unique_labels:
        active_domain = submodel_labels.where(submodel_labels.values == label_id)
        active_domain = ones_like(active_domain).where(active_domain.notnull(), 0)
        active_domain = active_domain.astype(submodel_labels.dtype)

        if submodel_label_to_mpi_rank is not None:
            if label_id not in submodel_label_to_mpi_rank:
                raise ValueError(
                    f"Label {label_id} not found in the MPI process mapping."
                )
            mpi_rank = submodel_label_to_mpi_rank[label_id]
            if mpi_rank < 0:
                raise ValueError(f"Negative MPI rank of {mpi_rank} is not allowed.")
        else:
            mpi_rank = -1

        submodel_partition_info = PartitionInfo(
            id=label_id, mpi_rank=mpi_rank, active_domain=active_domain
        )
        partition_infos.append(submodel_partition_info)

    return partition_infos


def _validate_submodel_label_array(submodel_labels: GridDataArray) -> None:
    unique_labels = np.unique(submodel_labels.values)

    if not (
        len(unique_labels) == unique_labels.max() + 1
        and unique_labels.min() == 0
        and np.issubdtype(submodel_labels.dtype, np.integer)
    ):
        raise ValueError(
            "The submodel_label  array should be integer and contain all the numbers between 0 and the number of "
            "partitions minus 1."
        )
