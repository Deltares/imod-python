from typing import List, NamedTuple

import numpy as np

from imod.mf6.model import GroundwaterFlowModel, Modflow6Model
from imod.mf6.utilities.clip import clip_by_grid
from imod.mf6.utilities.grid import get_active_domain_slice
from imod.typing import GridDataArray
from imod.typing.grid import ones_like, is_unstructured


class PartitionInfo(NamedTuple):
    active_domain: GridDataArray
    id: int


def create_partition_info(submodel_labels: GridDataArray) -> List[PartitionInfo]:
    """
    A PartitionInfo is used to partition a model or package. The partition info's of a domain are created using a
    submodel_labels array. The submodel_labels provided as input should have the same shape as a single layer of the
    model grid (all layers are split the same way), and contains an integer value in each cell. Each cell in the
    model grid will end up in the submodel with the index specified by the corresponding label of that cell. The
    labels should be numbers between 0 and the number of partitions.
    """
    _validate_submodel_label_array(submodel_labels)

    unique_labels = np.unique(submodel_labels.values)

    partition_infos = []
    for label_id in unique_labels:
        active_domain = submodel_labels.where(submodel_labels.values == label_id)
        active_domain = ones_like(active_domain).where(active_domain.notnull(), 0)
        active_domain = active_domain.astype(submodel_labels.dtype)

        submodel_partition_info = PartitionInfo(
            id=label_id, active_domain=active_domain
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


def slice_model(partition_info: PartitionInfo, model: Modflow6Model) -> Modflow6Model:
    """
    This function slices a Modflow6Model. A sliced model is a model that consists of packages of the original model
    that are sliced using the domain_slice. A domain_slice can be created using the
    :func:`imod.mf6.modelsplitter.create_domain_slices` function.
    """
    new_model = GroundwaterFlowModel(**model._options)
    if is_unstructured(model.domain):
        sliced_domain_layered = partition_info.active_domain.sel(        
            domain_slice2d
        ).broadcast_like(model.domain.layer)  
    else:
        domain_slice2d = get_active_domain_slice(partition_info.active_domain)
        sliced_domain_layered = partition_info.active_domain.sel(
            domain_slice2d
        ).broadcast_like(model.domain.layer)

    sliced_bottom = model.bottom
    if "dx" in sliced_domain_layered.coords:
        sliced_domain_layered = sliced_domain_layered.drop_vars("dx")
    if "dy" in sliced_domain_layered.coords:
        sliced_domain_layered = sliced_domain_layered.drop_vars("dy")
    if is_unstructured(model.domain):
        new_idomain = sliced_domain_layered
    else:
        new_idomain = model.domain.sel(sliced_domain_layered.coords).where(
            sliced_domain_layered, other=0
        )

    for pkg_name, package in model.items():
        sliced_package = clip_by_grid(package, partition_info.active_domain)

        sliced_package = sliced_package.mask(new_idomain)
        errors = sliced_package._validate(
            package._write_schemata, idomain=new_idomain, bottom=sliced_bottom
        )
        if not errors:
            new_model[pkg_name] = sliced_package
        else:
            print(f"package {pkg_name} removed")

    return new_model
