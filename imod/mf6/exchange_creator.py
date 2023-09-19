import numpy as np
import xarray as xr

from imod.mf6.gwfgwf import GWFGWF
from imod.mf6.utilities.grid_utilities import get_active_domain_slice


class ExchangeCreator:
    def __init__(self, submodel_labels, partition_info):
        self._connected_cells = find_connected_cells(submodel_labels)
        self._global_to_local_mapping = create_global_cellidx_to_local_cellid_mapping(
            partition_info
        )

    def create_exchanges(self, model_name, layers):
        self._connected_cells["layer"] = layers

        exchanges = []
        for model_id1, model_connection1 in self._connected_cells.groupby(
            "cell_label1"
        ):
            for model_id2, model_connection2 in model_connection1.groupby(
                "cell_label2"
            ):
                indices1 = np.nonzero(
                    np.isin(
                        self._global_to_local_mapping[model_id1]["global_cell_idx"],
                        model_connection2["cell_idx1"],
                    )
                )[0]
                indices2 = np.nonzero(
                    np.isin(
                        self._global_to_local_mapping[model_id2]["global_cell_idx"],
                        model_connection2["cell_idx2"],
                    )
                )[0]

                cell_ids = xr.Dataset()
                cell_ids["cell_id1"] = xr.DataArray(
                    self._global_to_local_mapping[model_id1]
                    .isel(connection_index=indices1)["local_cell_id"]
                    .values,
                    coords={
                        "connection_index": np.arange(len(indices1)),
                        "cell_dims1": ["row_1", "column_1"],
                    },
                )

                cell_ids["cell_id2"] = xr.DataArray(
                    self._global_to_local_mapping[model_id2]
                    .isel(connection_index=indices2)["local_cell_id"]
                    .values,
                    coords={
                        "connection_index": np.arange(len(indices2)),
                        "cell_dims2": ["row_2", "column_2"],
                    },
                )

                cell_ids = cell_ids.assign_coords({"layer": layers})

                cell_ids = (
                    cell_ids.stack(
                        cell_id=("layer", "connection_index"), create_index=False
                    )
                    .drop_vars("connection_index")
                    .reset_coords()
                )

                exchanges.append(
                    GWFGWF(
                        f"{model_name}_{model_id1}",
                        f"{model_name}_{model_id2}",
                        cell_ids["cell_id1"],
                        cell_ids["cell_id2"],
                        cell_ids["layer"],
                    )
                )

        return exchanges


def to_cell_idx(idomain):
    index = np.arange(idomain.size).reshape(idomain.shape)
    domain_index = xr.zeros_like(idomain)
    domain_index.values = index

    return domain_index


def find_connected_cells(submodel_labels):
    cell_indices = to_cell_idx(submodel_labels)

    diff_y1 = submodel_labels.diff("y", label="lower")
    diff_y2 = submodel_labels.diff("y", label="upper")

    connected_cells_idx_y1 = cell_indices.where(diff_y1 != 0, drop=True).astype(int)
    connected_cells_idx_y2 = cell_indices.where(diff_y2 != 0, drop=True).astype(int)

    connected_model_label1 = submodel_labels.where(diff_y1 != 0, drop=True).astype(int)
    connected_model_label2 = submodel_labels.where(diff_y2 != 0, drop=True).astype(int)

    connected_cell_info = xr.Dataset()
    connected_cell_info["cell_idx1"] = xr.DataArray(
        connected_cells_idx_y1.values.flatten(),
        coords={
            "connection_index": np.arange(connected_cells_idx_y2.size),
        },
    )

    connected_cell_info["cell_idx2"] = xr.DataArray(
        connected_cells_idx_y2.values.flatten(),
        coords={
            "connection_index": np.arange(connected_cells_idx_y2.size),
        },
    )

    connected_cell_info["cell_label1"] = xr.DataArray(
        connected_model_label1.values.flatten(),
        coords={
            "connection_index": np.arange(connected_cells_idx_y2.size),
        },
    )

    connected_cell_info["cell_label2"] = xr.DataArray(
        connected_model_label2.values.flatten(),
        coords={
            "connection_index": np.arange(connected_cells_idx_y2.size),
        },
    )

    return connected_cell_info


def create_cellid(row, column):
    return xr.DataArray(
        np.array([row.flatten() + 1, column.flatten() + 1]).T,
        coords={
            "connection_index": np.arange(row.size),
            "cell_dims": ["row", "column"],
        },
    )


def create_global_cellidx_to_local_cellid_mapping(partition_info):
    global_domain_cell_indices = to_cell_idx(partition_info[0].active_domain)

    mapping = {}

    for submodel_partition_info in partition_info:
        model_id = submodel_partition_info.id
        domain_slice = get_active_domain_slice(submodel_partition_info.active_domain)
        active = submodel_partition_info.active_domain.sel(domain_slice)

        local_domain_cell_indices = to_cell_idx(active)
        overlap = xr.merge(
            (global_domain_cell_indices, local_domain_cell_indices),
            join="inner",
            fill_value=np.nan,
            compat="override",
        )["idomain"]

        local_row, local_column = np.unravel_index(
            local_domain_cell_indices, local_domain_cell_indices.shape
        )

        mapping[model_id] = xr.Dataset()
        mapping[model_id]["global_cell_idx"] = xr.DataArray(
            overlap.values.flatten(),
            coords={
                "connection_index": np.arange(overlap.size),
            },
        )
        mapping[model_id]["local_cell_id"] = create_cellid(local_row, local_column)

    return mapping
