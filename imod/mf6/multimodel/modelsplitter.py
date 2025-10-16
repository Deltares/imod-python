from typing import List, NamedTuple

import numpy as np

from imod.common.interfaces.iagnosticpackage import IAgnosticPackage
from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.ipackage import IPackage
from imod.common.utilities.clip import clip_by_grid
from imod.mf6.boundary_condition import BoundaryCondition
from imod.typing import GridDataArray
from imod.typing.grid import (
    get_non_spatial_dimension_names,
    ones_like,
)


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


class ModelSplitter:
    # pkg_id to variable mapping.
    # For boundary packages we need to check if the package has any active cells in the partition
    # We do this based on the variable that defines the active cells for that package
    # This mapping is used to get that variable name based on the package id
    # If a package is not in this mapping, we assume it does not need special treatment
    _pkg_id_to_var_mapping = {
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

    # Some boundary packages don't have a variable that defines active cells
    # For these packages we skip the check if the package has any active cells in the partition
    _pkg_id_skip_active_domain_check = ["ssm", "lak"]

    def __init__(self, partition_info: List[PartitionInfo]) -> None:
        self.partition_info = partition_info

    def split(self, model_name: str, model: IModel) -> dict[str, IModel]:
        """
        Split a model into multiple partitioned models based on configured partition information.
        Parameters
        ----------
        model_name : str
            Base name of the input model; partition identifiers are appended to create
            names for each resulting submodel.
        model : IModel
            The input model instance to partition.
        Returns
        -------
        dict[str, IModel]
            A mapping from generated submodel names to the corresponding partitioned
            model instances, each containing only the packages and data relevant to its
            active domain.
        """
        modelclass = type(model)
        partitioned_models = {}
        model_to_partition = {}

        self._force_load_dis(model)

        # Create empty model for each partition
        for submodel_partition_info in self.partition_info:
            new_model_name = f"{model_name}_{submodel_partition_info.id}"

            new_model = modelclass(**model.options)
            partitioned_models[new_model_name] = new_model
            model_to_partition[new_model_name] = submodel_partition_info

        # Add packages to models
        for pkg_name, package in model.items():
            # Determine active domain for boundary packages
            active_package_domain = (
                self._get_package_domain(package)
                if isinstance(package, BoundaryCondition)
                else None
            )

            # Add package to each partitioned model
            for new_model_name, new_model in partitioned_models.items():
                partition_info = model_to_partition[new_model_name]

                has_overlap = self._has_package_data_in_domain(
                    package, active_package_domain, partition_info
                )
                if not has_overlap:
                    continue

                # Slice and add the package to the partitioned model
                sliced_package = clip_by_grid(package, partition_info.active_domain)

                # For agnostic packages, if the sliced package has no data, do not add it to the model
                if isinstance(package, IAgnosticPackage):
                    if sliced_package["index"].size == 0:
                        sliced_package = None

                # Add package to model if it has data
                if sliced_package is not None:
                    new_model[pkg_name] = sliced_package

        return partitioned_models

    def _force_load_dis(self, model) -> None:
        key = model.get_diskey()
        model[key].dataset.load()
        return

    def _get_package_domain(self, package: IPackage) -> GridDataArray | None:
        pkg_id = package.pkg_id
        active_package_domain = None

        if isinstance(package, BoundaryCondition):
            # Checks are done after slicing
            if isinstance(package, IAgnosticPackage):
                pass
            # No checks are done for these packages
            elif pkg_id in self._pkg_id_skip_active_domain_check:
                pass
            else:
                ds = package[self._pkg_id_to_var_mapping[pkg_id]]

                # Drop non-spatial dimensions and layer dimension if present
                dims_to_be_removed = get_non_spatial_dimension_names(ds)
                if "layer" in ds.dims:
                    dims_to_be_removed.append("layer")
                ds = ds.drop_vars(dims_to_be_removed)

                active_package_domain = ds.notnull()

        return active_package_domain

    def _has_package_data_in_domain(
        self,
        package: IPackage,
        active_package_domain: GridDataArray,
        partition_info: PartitionInfo,
    ) -> bool:
        pkg_id = package.pkg_id
        has_overlap = True
        if isinstance(package, BoundaryCondition):
            # Checks are done after slicing
            if isinstance(package, IAgnosticPackage):
                pass
            # No checks are done for these packages
            elif pkg_id in self._pkg_id_skip_active_domain_check:
                pass
            else:
                has_overlap = (
                    active_package_domain & partition_info.active_domain.astype(bool)
                ).any()  # type: ignore

        return has_overlap
