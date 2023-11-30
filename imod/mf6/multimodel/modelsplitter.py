from typing import List, NamedTuple

import numpy as np

from imod.mf6.hfb import HorizontalFlowBarrierBase
from imod.mf6.model import GroundwaterFlowModel, Modflow6Model
from imod.mf6.utilities.clip import clip_by_grid
from imod.mf6.utilities.grid import get_active_domain_slice
from imod.mf6.utilities.schemata import filter_schemata_dict
from imod.mf6.wel import Well
from imod.schemata import AllNoDataSchema
from imod.typing import GridDataArray
from imod.typing.grid import is_unstructured, ones_like

HIGH_LEVEL_PKGS = (HorizontalFlowBarrierBase, Well)


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
    domain_slice2d = get_active_domain_slice(partition_info.active_domain)
    if is_unstructured(model.domain):
        new_idomain = model.domain.sel(domain_slice2d)
    else:
        # store the original layer dimension
        layer = model.domain.layer

        sliced_domain_2D = partition_info.active_domain.sel(domain_slice2d)
        # drop the dimensions we don't need from the 2D domain slice. It may have a single
        # layer so we drop that as well
        sliced_domain_2D = sliced_domain_2D.drop_vars(
            ["dx", "dy", "layer"], errors="ignore"
        )
        # create the desired coodinates: the original layer dimension,and the x/y coordinates of the slice.
        coords = dict(layer=layer, **sliced_domain_2D.coords)

        # the new idomain is the selection on our coodinates and only the part active in sliced_domain_2D
        new_idomain = model.domain.sel(coords).where(sliced_domain_2D, other=0)

    for pkg_name, package in model.items():
        sliced_package = clip_by_grid(package, partition_info.active_domain)

        sliced_package = sliced_package.mask(new_idomain)
        # The masking can result in packages with AllNoData.Therefore we'll have
        # to drop these packages. Create schemata dict only containing the
        # variables with a AllNoDataSchema.
        allnodata_schemata = filter_schemata_dict(
            package._write_schemata, (AllNoDataSchema)
        )
        # Find if packages throws ValidationError for AllNoDataSchema.
        allnodata_errors = sliced_package._validate(allnodata_schemata)
        # Drop if allnodata error thrown
        if not allnodata_errors:
            new_model[pkg_name] = sliced_package
        else:
            # TODO: Add this to logger
            print(
                f"package {pkg_name} removed in partition {partition_info.id}, because all empty"
            )

    return new_model
