import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.gwfgwf import GWFGWF
from imod.mf6.utilities.grid_utilities import get_active_domain_slice


class ExchangeCreator:
    def __init__(self, submodel_labels, partition_info):
        self._submodel_labels = submodel_labels

        self._global_cell_indices = _to_cell_idx(submodel_labels)
        self._connected_cells = self._find_connected_cells()
        self._global_to_local_mapping = (
            self._create_global_cellidx_to_local_cellid_mapping(partition_info)
        )

    def create_exchanges(self, model_name, layers):
        layers = layers.to_dataframe()

        exchanges = []
        for model_id1, grouped_connected_models in self._connected_cells.groupby(
            "cell_label1"
        ):
            for model_id2, connected_domain_pair in grouped_connected_models.groupby(
                "cell_label2"
            ):
                mapping1 = (
                    self._global_to_local_mapping[model_id1]
                    .drop(columns=["local_idx"])
                    .rename(
                        columns={"global_idx": "cell_idx1", "local_cell_id": "cell_id1"}
                    )
                )

                mapping2 = (
                    self._global_to_local_mapping[model_id2]
                    .drop(columns=["local_idx"])
                    .rename(
                        columns={"global_idx": "cell_idx2", "local_cell_id": "cell_id2"}
                    )
                )

                connected_cells = (
                    connected_domain_pair.merge(mapping1)
                    .merge(mapping2)
                    .filter(["cell_id1", "cell_id2"])
                )

                connected_cells = pd.merge(layers, connected_cells, how="cross")

                exchanges.append(
                    GWFGWF(
                        f"{model_name}_{model_id1}",
                        f"{model_name}_{model_id2}",
                        connected_cells["cell_id1"].values,
                        connected_cells["cell_id2"].values,
                        connected_cells["layer"].values,
                    )
                )

        return exchanges

    def _find_connected_cells(self):
        connected_cells_along_x = self._find_connected_cells_along_axis("x")
        connected_cells_along_y = self._find_connected_cells_along_axis("y")

        return pd.merge(connected_cells_along_x, connected_cells_along_y, how="outer")

    def _find_connected_cells_along_axis(self, axis_label):
        diff1 = self._submodel_labels.diff(f"{axis_label}", label="lower")
        diff2 = self._submodel_labels.diff(f"{axis_label}", label="upper")

        connected_cells_idx1 = self._global_cell_indices.where(
            diff1 != 0, drop=True
        ).astype(int)
        connected_cells_idx2 = self._global_cell_indices.where(
            diff2 != 0, drop=True
        ).astype(int)

        connected_model_label1 = self._submodel_labels.where(
            diff1 != 0, drop=True
        ).astype(int)
        connected_model_label2 = self._submodel_labels.where(
            diff2 != 0, drop=True
        ).astype(int)

        connected_cell_info = pd.DataFrame(
            {
                "cell_idx1": connected_cells_idx1.values.flatten(),
                "cell_idx2": connected_cells_idx2.values.flatten(),
                "cell_label1": connected_model_label1.values.flatten(),
                "cell_label2": connected_model_label2.values.flatten(),
            }
        )

        return connected_cell_info

    def _create_global_cellidx_to_local_cellid_mapping(self, partition_info):
        global_to_local_idx = _create_global_to_local_idx(
            partition_info, self._global_cell_indices
        )
        local_cell_idx_to_id = _local_cell_idx_to_id(partition_info)

        mapping = {}
        for submodel_partition_info in partition_info:
            model_id = submodel_partition_info.id
            mapping[model_id] = pd.merge(
                global_to_local_idx[model_id], local_cell_idx_to_id[model_id]
            )

        return mapping


def _create_global_to_local_idx(partition_info, global_cell_indices):
    global_to_local_idx = {}
    for submodel_partition_info in partition_info:
        local_cell_indices = _get_local_cell_indices(submodel_partition_info)
        overlap = xr.merge(
            (global_cell_indices, local_cell_indices),
            join="inner",
            fill_value=np.nan,
            compat="override",
        )["idomain"]

        model_id = submodel_partition_info.id
        global_to_local_idx[model_id] = pd.DataFrame(
            {
                "global_idx": overlap.values.flatten(),
                "local_idx": local_cell_indices.values.flatten(),
            }
        )

    return global_to_local_idx


def _local_cell_idx_to_id(partition_info):
    local_cell_idx_to_id = {}
    for submodel_partition_info in partition_info:
        local_cell_indices = _get_local_cell_indices(submodel_partition_info)
        local_row, local_column = np.unravel_index(
            local_cell_indices, local_cell_indices.shape
        )

        model_id = submodel_partition_info.id
        local_cell_idx_to_id[model_id] = pd.DataFrame(
            {
                "local_idx": local_cell_indices.values.flatten(),
                "local_cell_id": zip(
                    local_row.flatten() + 1, local_column.flatten() + 1
                ),
            }
        )

    return local_cell_idx_to_id


def _get_local_cell_indices(submodel_partition_info):
    domain_slice = get_active_domain_slice(submodel_partition_info.active_domain)
    local_domain = submodel_partition_info.active_domain.sel(domain_slice)

    return _to_cell_idx(local_domain)


def _to_cell_idx(idomain):
    index = np.arange(idomain.size).reshape(idomain.shape)
    domain_index = xr.zeros_like(idomain)
    domain_index.values = index

    return domain_index
