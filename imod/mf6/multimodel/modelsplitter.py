from typing import List, NamedTuple

import numpy as np

from imod.common.interfaces.iagnosticpackage import IAgnosticPackage
from imod.common.interfaces.imodel import IModel
from imod.common.utilities.clip import clip_by_grid
from imod.mf6.auxiliary_variables import (
    expand_transient_auxiliary_variables,
    remove_expanded_auxiliary_variables_from_dataset,
)
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.hfb import HorizontalFlowBarrierBase
from imod.mf6.wel import Well
from imod.typing import GridDataArray
from imod.typing.grid import ones_like

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


def slice_model(partition_info: PartitionInfo, model: IModel) -> IModel:
    """
    This function slices a Modflow6Model. A sliced model is a model that
    consists of packages of the original model that are sliced using the
    domain_slice. A domain_slice can be created using the
    :func:`imod.mf6.modelsplitter.create_domain_slices` function.
    """
    modelclass = type(model)
    new_model = modelclass(**model.options)

    for pkg_name, package in model.items():
        if isinstance(package, BoundaryCondition):
            remove_expanded_auxiliary_variables_from_dataset(package)

        sliced_package = clip_by_grid(package, partition_info.active_domain)
        if sliced_package is not None:
            new_model[pkg_name] = sliced_package

        if isinstance(package, BoundaryCondition):
            expand_transient_auxiliary_variables(sliced_package)
    return new_model


class ModelSplitter:
    def __init__(self, partition_info: List[PartitionInfo]) -> None:
        self.partition_info = partition_info

    def split(self, model_name: str, model: IModel) -> dict[str, IModel]:
        modelclass = type(model)
        partitioned_models = {}
        model_to_partition = {}

        # Create empty model for each partition
        for submodel_partition_info in self.partition_info:
            new_model_name = f"{model_name}_{submodel_partition_info.id}"

            new_model = modelclass(**model.options)
            partitioned_models[new_model_name] = new_model
            model_to_partition[new_model_name] = submodel_partition_info

        # Create pkg_id to variable mapping
        # We don't want to add packages that do not have any active cells in the partition
        # We determine if a package has active cells in the partition based on the variable
        # that defines the active cells for that package
        pkg_id_to_var_mapping = {
            "chd": "head",
            "cnc": "concentration",
            "evt": "rate",
            "dis": "idomain",
            "drn": "elevation",
            "ghb": "head",
            "src": "rate",
            "rch": "rate",
            "riv": "conductance",
            "uzf": "infiltration_rate",
            "wel": "rate",
        }
        # Add packages to models
        for pkg_name, package in model.items():
            pkg_id = package._pkg_id

            # Determine the active cells of boundary package
            if isinstance(package, IAgnosticPackage):
                pass
            elif pkg_id in ["ssm", "lak"]:
                pass
            elif isinstance(package, BoundaryCondition):
                active_package_domain = package[pkg_id_to_var_mapping[pkg_id]].notnull()
            else:
                pass  # Other packages don't need special treatment

            for new_model_name, new_model in partitioned_models.items():
                partition_info = model_to_partition[new_model_name]

                # Check if package has any active cells in the partition
                # If not, skip adding the package to the partitioned model
                if isinstance(package, IAgnosticPackage):
                    pass
                elif pkg_id in ["ssm", "lak"]:  # No checks are done for these packages
                    pass
                elif isinstance(package, BoundaryCondition):
                    has_overlap = (
                        active_package_domain & (partition_info.active_domain == 1)
                    ).any()
                    if not has_overlap:
                        continue

                # Slice and add the package to the partitioned model
                sliced_package = clip_by_grid(package, partition_info.active_domain)

                if isinstance(package, IAgnosticPackage):
                    if sliced_package["index"].size == 0:
                        sliced_package = None

                if sliced_package is not None:
                    new_model[pkg_name] = sliced_package

        return partitioned_models

    def num_partitions(self) -> int:
        return len(self.partition_info)
