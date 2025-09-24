import collections
from typing import Any, NamedTuple

import numpy as np
from plum import Dispatcher

import imod
from imod.common.interfaces.ilinedatapackage import ILineDataPackage
from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.ipackagebase import IPackageBase
from imod.common.interfaces.ipointdatapackage import IPointDataPackage
from imod.common.utilities.clip import clip_by_grid
from imod.mf6.auxiliary_variables import (
    expand_transient_auxiliary_variables,
    remove_expanded_auxiliary_variables_from_dataset,
)
from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.hfb import HorizontalFlowBarrierBase
from imod.mf6.multimodel.exchange_creator import PartitionInfo
from imod.mf6.multimodel.exchange_creator_structured import ExchangeCreator_Structured
from imod.mf6.multimodel.exchange_creator_unstructured import (
    ExchangeCreator_Unstructured,
)
from imod.mf6.wel import Well
from imod.typing import GridDataArray
from imod.typing.grid import bounding_polygon, is_unstructured

HIGH_LEVEL_PKGS = (HorizontalFlowBarrierBase, Well)


dispatch = Dispatcher()


@dispatch
def activity_count(
    package: object, labels: object, polygons: list[Any], ignore_time_purge_empty: bool
) -> dict:
    raise TypeError(
        f"`labels` should be of type xr.DataArray, xu.Ugrid2d or xu.UgridDataArray, got {type(labels)}"
    )


@dispatch
def activity_count(  # noqa: F811
    package: IPackageBase,
    labels: object,
    polygons: list[Any],
    ignore_time_purge_empty: bool,
) -> dict:
    label_dims = set(labels.dims)
    dataset = package.dataset

    # Determine sample variable: it should be spatial.
    # Otherwise return a count of 1 for each partition.
    if not label_dims.intersection(dataset.dims):
        return dict.fromkeys(range(len(polygons)), 1)

    # Find variable with spatial dimensions
    # Accessing variables is cheaper than creating a DataArray.
    ndim_per_variable = {
        var_name: len(dims)
        for var_name in dataset.data_vars
        if label_dims.intersection(dims := dataset.variables[var_name].dims)
    }
    max_variable = max(ndim_per_variable, key=ndim_per_variable.get)
    # TODO: there might be a more robust way to do this.
    # Alternatively, we just define a predicate variable (e.g. conductance)
    # on each package.

    sample = dataset[max_variable]
    if "time" in sample.coords:
        if ignore_time_purge_empty:
            sample = sample.isel(time=0)
        else:
            sample = sample.max("time")

    # Use ellipsis to reduce over ALL dimensions except label dims
    dims_to_aggregate = [dim for dim in sample.dims if dim not in label_dims]
    counts = sample.notnull().sum(dim=dims_to_aggregate).groupby(labels).sum()
    return {label: int(n) for label, n in enumerate(counts.data)}


@dispatch
def activity_count(  # noqa: F811
    package: IPointDataPackage,
    labels: object,
    polygons: list[Any],
    ignore_time_purge_empty: bool,
) -> dict:
    point_labels = imod.select.points_values(
        labels, out_of_bounds="ignore", x=package.x, y=package.y
    )
    return {label: int(n) for label, n in enumerate(np.bincount(point_labels))}


@dispatch
def activity_count(  # noqa: F811
    package: ILineDataPackage,
    labels: object,
    polygons: list[Any],
    ignore_time_purge_empty: bool,
) -> dict:
    counts = {}
    gdf_linestrings = package.line_data
    for partition_id, polygon in enumerate(polygons):
        partition_linestrings = gdf_linestrings.clip(polygon)
        # Catch edge case: when line crosses only vertex of polygon, a point
        # or multipoint is returned. These will be dropped, and can be
        # identified by zero length.
        counts[partition_id] = sum(partition_linestrings.length > 0)
    return counts


class PartitionModels(NamedTuple):
    """
    Mapping of:
        flow_model_name (str) => model (object)
        partition_id (int) => transport_model_name (str) => model (object)
    """

    flow_models: dict[str, object]
    transport_models: dict[int, dict[str, object]]

    def paired_keys(self):
        for partition_id, key in enumerate(self.flow_models.keys()):
            yield key, list(self.transport_models[partition_id].keys())

    def paired_models(self):
        for partition_id, model in enumerate(self.flow_models.values()):
            yield model, list(self.transport_models[partition_id].values())

    def paired_items(self):
        for partition_id, (key, model) in enumerate(self.flow_models.items()):
            partition_models = self.transport_models[partition_id]
            yield (
                (key, model),
                (list(partition_models.keys()), list(partition_models.values())),
            )

    @property
    def flat_transport_models(self):
        return {
            name: model
            for partition_models in self.transport_models.values()
            for name, model in partition_models.items()
        }


class ModelSplitter:
    def __init__(
        self,
        flow_models: dict[str, object],
        transport_models: dict[str, object],
        submodel_labels: GridDataArray,
        ignore_time_purge_empty: bool = False,
    ):
        self.flow_models = flow_models
        self.transport_models = transport_models
        self.models = {**flow_models, **transport_models}
        self.submodel_labels = submodel_labels
        self.unique_labels = self._validate_submodel_label_array(submodel_labels)
        self.ignore_time_purge_empty = ignore_time_purge_empty
        self._create_partition_info()
        self.bounding_polygons = [
            bounding_polygon(partition.active_domain)
            for partition in self.partition_info
        ]

        self.exchange_creator: ExchangeCreator_Unstructured | ExchangeCreator_Structured
        if is_unstructured(self.submodel_labels):
            self.exchange_creator = ExchangeCreator_Unstructured(
                self.submodel_labels, self.partition_info
            )
        else:
            self.exchange_creator = ExchangeCreator_Structured(
                self.submodel_labels, self.partition_info
            )

        self._count_boundary_activity_per_partition()

    @property
    def modelnames(self):
        return list(self.models.keys())

    @staticmethod
    def _validate_submodel_label_array(submodel_labels: GridDataArray) -> None:
        unique_labels = np.unique(submodel_labels)

        if not (
            len(unique_labels) == unique_labels.max() + 1
            and unique_labels.min() == 0
            and np.issubdtype(submodel_labels.dtype, np.integer)
        ):
            raise ValueError(
                "The submodel_label array should be integer and contain all the numbers between 0 and the number of "
                "partitions minus 1."
            )
        return unique_labels

    def _create_partition_info(self):
        self.partition_info = []
        labels = self.submodel_labels
        for label_id in self.unique_labels:
            active_domain = (labels == label_id).astype(labels.dtype)
            self.partition_info.append(
                PartitionInfo(
                    active_domain=active_domain,
                    partition_id=int(label_id),
                )
            )

    def _create_partition_polygons(self):
        self.partition_polygons = {
            info.partition_id: bounding_polygon(info.active_domain)
            for info in self.partition_info
        }

    def _count_boundary_activity_per_partition(self):
        counts = {}
        for model_name, model in self.models.items():
            model_counts = {}
            for pkg_name, package in model.items():
                # Packages like NPF, DIS are always required.
                # We only need to check packages with a MAXBOUND entry.
                if not isinstance(package, BoundaryCondition):
                    continue
                model_counts[pkg_name] = activity_count(
                    package,
                    self.submodel_labels,
                    self.bounding_polygons,
                    self.ignore_time_purge_empty,
                )
            counts[model_name] = model_counts
        self.boundary_activity_counts = counts

    def slice_model(
        self, model: IModel, info: PartitionInfo, boundary_activity_counts: dict
    ) -> IModel:
        modelclass = type(model)
        new_model = modelclass(**model.options)

        for pkg_name, package in model.items():
            if isinstance(package, BoundaryCondition):
                # Skip empty boundary conditions
                if boundary_activity_counts[pkg_name][info.partition_id] == 0:
                    continue
                else:
                    remove_expanded_auxiliary_variables_from_dataset(package)

            sliced_package = clip_by_grid(package, info.active_domain)
            if sliced_package is not None:
                new_model[pkg_name] = sliced_package

            if isinstance(package, BoundaryCondition):
                expand_transient_auxiliary_variables(sliced_package)

        return new_model

    def _split(self, models, nest: bool):
        partition_models = collections.defaultdict(dict)
        for model_name, model in models.items():
            for info in self.partition_info:
                new_model = self.slice_model(
                    model, info, self.boundary_activity_counts[model_name]
                )
                new_model_name = f"{model_name}_{info.partition_id}"
                if nest:
                    partition_models[info.partition_id][new_model_name] = new_model
                else:
                    partition_models[new_model_name] = new_model
        return partition_models

    def split(self):
        # FUTURE: we may currently assume there is a single flow model. See check above.
        # And each separate transport model represents a different species.
        flow_models = self._split(self.flow_models, nest=False)
        transport_models = self._split(self.transport_models, nest=True)
        return PartitionModels(flow_models, transport_models)

    def create_gwfgwf_exchanges(self):
        exchanges: list[Any] = []
        for model_name, model in self.flow_models.items():
            exchanges += self.exchange_creator.create_gwfgwf_exchanges(
                model_name, model.domain.layer
            )
        return exchanges

    def create_gwtgwt_exchanges(self):
        exchanges: list[Any] = []
        # TODO: weird/arbitrary dependence on the single flow model?
        flow_model_name = list(self.flow_models.keys())[0]
        model = self.flow_models[flow_model_name]
        if any(self.transport_models):
            for transport_model_name in self.transport_models:
                exchanges += self.exchange_creator.create_gwtgwt_exchanges(
                    transport_model_name, flow_model_name, model.domain.layer
                )
        return exchanges
