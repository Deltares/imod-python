from typing import List, NamedTuple

import numpy as np

from imod.common.interfaces.iagnosticpackage import IAgnosticPackage
from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.ipackage import IPackage
from imod.common.utilities.clip import clip_by_grid
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.ims import Solution
from imod.mf6.model_gwf import GroundwaterFlowModel
from imod.mf6.model_gwt import GroundwaterTransportModel
from imod.mf6.ssm import SourceSinkMixing
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

        # Initialize mapping from original model names to partitioned models
        self._model_to_partitioned_model: dict[str, dict[str, IModel]] = {}

        # Initialize mapping from partition IDs to models
        self._partition_id_to_models: dict[int, dict[str, IModel]] = {}
        for submodel_partition_info in self.partition_info:
            self._partition_id_to_models[submodel_partition_info.id] = {}

    def split(self, model_name: str, model: IModel) -> dict[str, IModel]:
        """
        Split a model into multiple partitioned models based on partition information.

        Each partition creates a separate submodel containing:
        - All non-boundary packages from the original model, clipped to the partition's domain
        - Boundary packages that have active cells within the partition's domain, clipped accordingly
        - IAgnosticPackages are excluded if they contain no data after clipping

        Parameters
        ----------
        model_name : str
            Base name of the input model. Partition IDs are appended to create
            unique names for each submodel (e.g., "model_0", "model_1").
        model : IModel
            The input model instance to partition.

        Returns
        -------
        dict[str, IModel]
            A mapping from generated submodel names to the corresponding partitioned
            model instances, each clipped to its respective active domain.
        """
        modelclass = type(model)
        partitioned_models = {}
        model_to_partition = {}

        # Create empty model for each partition
        for submodel_partition_info in self.partition_info:
            new_model_name = f"{model_name}_{submodel_partition_info.id}"

            new_model = modelclass(**model.options)
            partitioned_models[new_model_name] = new_model
            model_to_partition[new_model_name] = submodel_partition_info
            self._partition_id_to_models[submodel_partition_info.id][new_model_name] = (
                new_model
            )

        self._model_to_partitioned_model[model_name] = partitioned_models

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

    def update_dependent_packages(self) -> None:
        """
        Update packages that reference other models after partitioning.

        This method performs two updates:
        1. Updates buoyancy packages in flow models to reference the correct
        partitioned transport model names.
        2. Recreates Source Sink Mixing (SSM) packages in transport models
        based on the partitioned flow model data.
        """
        # Update buoyancy packages
        for _, models in self._partition_id_to_models.items():
            flow_model = self._get_flow_model(models)
            transport_model_names = self._get_transport_model_names(models)

            flow_model._update_buoyancy_package(transport_model_names)

        # Update ssm packages
        for _, models in self._partition_id_to_models.items():
            flow_model = self._get_flow_model(models)
            transport_models = self._get_transport_models(models)

            for transport_model in transport_models:
                ssm_key = transport_model._get_pkgkey("ssm")
                if ssm_key is None:
                    continue
                old_ssm_package = transport_model.pop(ssm_key)
                state_variable_name = old_ssm_package.dataset[
                    "auxiliary_variable_name"
                ].values[0]
                ssm_package = SourceSinkMixing.from_flow_model(
                    flow_model, state_variable_name, is_split=True
                )
                if ssm_package is not None:
                    transport_model[ssm_key] = ssm_package

    def update_solutions(
        self, original_model_name_to_solution: dict[str, Solution]
    ) -> None:
        """
        Update solution objects to reference partitioned models instead of original models.

        For each original model that was split:
        1. Removes the original model reference from its solution (This was a deepcopy of the original solution and thus references the original model).
        2. Adds all partitioned submodel references to the same solution

        This ensures that the Solution objects correctly reference the new partitioned
        model names after splitting.
        """
        for model_name, new_models in self._model_to_partitioned_model.items():
            solution = original_model_name_to_solution[model_name]
            solution._remove_model_from_solution(model_name)
            for new_model_name, new_model in new_models.items():
                solution._add_model_to_solution(new_model_name)


    def _is_package_to_skip(self, package: IPackage) -> bool:
        """
        Determine if a package should be skipped in grid checks.

        Package can be skipped in the following cases:
        - Package is not a BoundaryCondition (non-boundary packages are always included)
        - Package is an IAgnosticPackage (overlap check deferred until after slicing)
        - Package is in _pkg_id_skip_active_domain_check (e.g., SSM, LAK packages)
        """
        pkg_id = package.pkg_id
        if not isinstance(package, BoundaryCondition):
            return True
        return isinstance(package, IAgnosticPackage) or (
            pkg_id in self._pkg_id_skip_active_domain_check
        )


    def _get_package_domain(self, package: IPackage) -> GridDataArray | None:
        """
        Extract the active domain of a boundary condition package.

        For boundary condition packages, this method identifies which cells contain
        active boundary data by checking the package's defining variable (e.g.,
        "head" for CHD, "rate" for WEL). Non-boundary packages return None.

        The active domain is determined by:
        1. Retrieving the variable that defines active cells (from _pkg_id_to_var_mapping)
        2. Removing non-spatial dimensions
        3. Creating a boolean mask where non-null values indicate active cells

        Returns None if the package is not a boundary condition or should be skipped.
        """
        if self._is_package_to_skip(package):
            return None

        pkg_id = package.pkg_id
        ds = package[self._pkg_id_to_var_mapping[pkg_id]]

        # Drop non-spatial dimensions if present
        dims_to_be_removed = get_non_spatial_dimension_names(ds)
        ds = ds.drop_vars(dims_to_be_removed)

        active_package_domain = ds.notnull()
        return active_package_domain

    def _has_package_data_in_domain(
        self,
        package: IPackage,
        active_package_domain: GridDataArray,
        partition_info: PartitionInfo,
    ) -> bool:
        """
        Check if a package has any active data within a partition's domain.

        For boundary condition packages, this method determines whether the package
        should be included in a partitioned model by checking if its active cells
        overlap with the partition's active domain.

        The method returns True in the following cases:
        - Package should be skipped in grid checks.
        - Package has at least one active cell overlapping with the partition domain
        """
        if self._is_package_to_skip(package):
            return True

        has_overlap = (
            active_package_domain & partition_info.active_domain.astype(bool)
        ).any()  # type: ignore

        return has_overlap

    def _get_flow_model(self, models: dict[str, IModel]) -> GroundwaterFlowModel:
        flow_model = next(
            (
                model
                for model_name, model in models.items()
                if isinstance(model, GroundwaterFlowModel)
            ),
            None,
        )

        if flow_model is None:
            raise ValueError(
                "Could not find a groundwater flow model for updating the buoyancy package."
            )

        return flow_model

    def _get_transport_model_names(self, models: dict[str, IModel]) -> List[str]:
        return [
            model_name
            for model_name, model in models.items()
            if isinstance(model, GroundwaterTransportModel)
        ]

    def _get_transport_models(
        self, models: dict[str, IModel]
    ) -> List[GroundwaterTransportModel]:
        return [
            model
            for model_name, model in models.items()
            if isinstance(model, GroundwaterTransportModel)
        ]
