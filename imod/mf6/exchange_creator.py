from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.gwfgwf import GWFGWF
from imod.mf6.modelsplitter import PartitionInfo
from imod.mf6.utilities.grid_utilities import (
    create_geometric_grid_info,
    get_active_domain_slice,
    to_cell_idx,
)
from imod.typing.grid import GridDataArray


class ExchangeCreator:
    """
    Creates the GroundWaterFlow to GroundWaterFlow exchange definitions as a function of a submodel label array and a
    PartitionInfo object. At the moment only structured grids are supported

    The submodel_labels array should have the same topology as the domain being partitioned. The array will be used
    to determine the connectivity of the submodels after the split operation has been performed.

    """

    def __init__(
        self, submodel_labels: GridDataArray, partition_info: List[PartitionInfo]
    ):
        self._submodel_labels = submodel_labels

        self._global_cell_indices = to_cell_idx(submodel_labels)
        self._connected_cells = self._find_connected_cells()
        self._global_to_local_mapping = (
            self._create_global_cellidx_to_local_cellid_mapping(partition_info)
        )
        self._geometric_information = self._compute_geometric_information()

    def create_exchanges(self, model_name: str, layers: GridDataArray) -> List[GWFGWF]:
        """
        Create GroundWaterFlow-GroundWaterFlow exchanges based on the submodel_labels array provided in the class
        constructor. The layer parameter is used to extrude the cell connection through all the layers. An exchange
        contains:
        - the model names of the connected submodel
        - the local cell id of the first model
        - the local cell id of the second model
        - the layer on which the connected cells reside

         For each connection between submodels only a single exchange is created. So if an exchange of model1 to
         model2 is created then no exchange for model2 to model1 will be created.

         Add the moment the geometric coefficients aren't computed.


        Returns
        -------
        a list of GWFGWF-exchanges

        """
        layers = layers.to_dataframe()

        connected_cells_with_geometric_info = pd.merge(
            self._connected_cells, self._geometric_information
        )

        exchanges = []
        for (
            model_id1,
            grouped_connected_models,
        ) in connected_cells_with_geometric_info.groupby("cell_label1"):
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
                    .filter(["cell_id1", "cell_id2", "cl1", "cl2", "hwva"])
                )

                connected_cells = pd.merge(layers, connected_cells, how="cross")

                exchanges.append(
                    GWFGWF(
                        f"{model_name}_{model_id1}",
                        f"{model_name}_{model_id2}",
                        connected_cells["cell_id1"].values,
                        connected_cells["cell_id2"].values,
                        connected_cells["layer"].values,
                        connected_cells["cl1"].values,
                        connected_cells["cl2"].values,
                        connected_cells["hwva"].values,
                    )
                )

        return exchanges

    def _find_connected_cells(self) -> pd.DataFrame:
        connected_cells_along_x = self._find_connected_cells_along_axis("x")
        connected_cells_along_y = self._find_connected_cells_along_axis("y")

        return pd.merge(connected_cells_along_x, connected_cells_along_y, how="outer")

    def _find_connected_cells_along_axis(self, axis_label: str) -> pd.DataFrame:
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

    def _create_global_cellidx_to_local_cellid_mapping(
        self, partition_info: List[PartitionInfo]
    ) -> Dict[int, pd.DataFrame]:
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

    def _compute_geometric_information(self) -> pd.DataFrame:
        geometric_grid_info = create_geometric_grid_info(self._global_cell_indices)

        cell1_df = geometric_grid_info.take(self._connected_cells["cell_idx1"])
        cell2_df = geometric_grid_info.take(self._connected_cells["cell_idx2"])

        distance_x = np.abs(cell1_df["x"].values - cell2_df["x"].values)
        distance_y = np.abs(cell1_df["y"].values - cell2_df["y"].values)
        distance = np.sqrt(distance_x**2 + distance_y**2)

        cl1 = 0.5 * np.where(
            distance_x > distance_y, cell1_df["dx"].values, cell1_df["dy"].values
        )
        cl2 = 0.5 * np.where(
            distance_x > distance_y, cell2_df["dx"].values, cell2_df["dy"].values
        )

        df = pd.DataFrame(
            {
                "cell_idx1": self._connected_cells["cell_idx1"].values,
                "cell_idx2": self._connected_cells["cell_idx2"].values,
                "cl1": cl1,
                "cl2": cl2,
                "hwva": distance.flatten(),
            }
        )

        return df


def _create_global_to_local_idx(
    partition_info: List[PartitionInfo], global_cell_indices: GridDataArray
) -> Dict[int, pd.DataFrame]:
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


def _local_cell_idx_to_id(partition_info) -> Dict[int, pd.DataFrame]:
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


def _get_local_cell_indices(submodel_partition_info: PartitionInfo) -> xr.DataArray:
    domain_slice = get_active_domain_slice(submodel_partition_info.active_domain)
    local_domain = submodel_partition_info.active_domain.sel(domain_slice)

    return to_cell_idx(local_domain)
