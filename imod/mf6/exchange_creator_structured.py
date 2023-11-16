from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.exchange_creator import ExchangeCreator
from imod.mf6.modelsplitter import PartitionInfo
from imod.mf6.utilities.grid import create_geometric_grid_info
from imod.typing import GridDataArray

NOT_CONNECTED_VALUE = -999


class ExchangeCreator_Structured(ExchangeCreator):
    """
    Creates the GroundWaterFlow to GroundWaterFlow exchange package (gwfgwf) as
    a function of a submodel label array and a PartitionInfo object. This file
    contains the cell indices of coupled cells. With coupled cells we mean cells
    that are adjacent but that are located in different subdomains. At the
    moment only structured grids are supported, for unstructured grids the
    geometric information is still set to default values.

    The submodel_labels array should have the same topology as the domain being partitioned. The array will be used
    to determine the connectivity of the submodels after the split operation has been performed.

    """

    def __init__(
        self, submodel_labels: GridDataArray, partition_info: List[PartitionInfo]
    ):
        super().__init__(submodel_labels, partition_info)

    @property
    def _coordinate_names(self):
        return ["row", "column"]

    def _find_connected_cells(self) -> pd.DataFrame:
        connected_cells_along_x = self._find_connected_cells_along_axis("x")
        connected_cells_along_y = self._find_connected_cells_along_axis("y")

        return pd.merge(connected_cells_along_x, connected_cells_along_y, how="outer")

    def _find_connected_cells_along_axis(self, axis_label: str) -> pd.DataFrame:
        diff1 = self._submodel_labels.diff(f"{axis_label}", label="lower")
        diff2 = self._submodel_labels.diff(f"{axis_label}", label="upper")

        connected_cells_idx1 = self._global_cell_indices.where(
            diff1 != 0, drop=True, other=NOT_CONNECTED_VALUE
        ).astype(int)
        connected_cells_idx2 = self._global_cell_indices.where(
            diff2 != 0, drop=True, other=NOT_CONNECTED_VALUE
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
        connected_cell_info = connected_cell_info.loc[
            connected_cell_info.cell_idx1 != NOT_CONNECTED_VALUE
        ]
        label_increasing = (
            connected_cell_info["cell_label1"] < connected_cell_info["cell_label2"]
        )

        connected_cell_info.loc[
            label_increasing, ["cell_idx1", "cell_idx2", "cell_label1", "cell_label2"]
        ] = connected_cell_info.loc[
            label_increasing, ["cell_idx2", "cell_idx1", "cell_label2", "cell_label1"]
        ].values

        return connected_cell_info

    def _compute_geometric_information(self) -> pd.DataFrame:
        geometric_grid_info = create_geometric_grid_info(self._global_cell_indices)

        cell1_df = geometric_grid_info.take(self._connected_cells["cell_idx1"])
        cell2_df = geometric_grid_info.take(self._connected_cells["cell_idx2"])

        distance_x = np.abs(cell1_df["x"].values - cell2_df["x"].values)
        distance_y = np.abs(cell1_df["y"].values - cell2_df["y"].values)

        is_x_connection = distance_x > distance_y
        cl1 = 0.5 * np.where(
            is_x_connection, cell1_df["dx"].values, cell1_df["dy"].values
        )
        cl2 = 0.5 * np.where(
            is_x_connection, cell2_df["dx"].values, cell2_df["dy"].values
        )
        hwva = np.where(is_x_connection, cell2_df["dy"].values, cell2_df["dx"].values)

        cdist = np.where(is_x_connection, distance_x, distance_y)

        outward_vector = np.zeros((len(is_x_connection), 2))
        outward_vector[:, 0] = cell2_df["x"].values - cell1_df["x"].values
        outward_vector[:, 1] = cell2_df["y"].values - cell1_df["y"].values
        anglex = np.arctan2(outward_vector[:, 1], outward_vector[:, 0])
        angledegx = np.degrees(anglex) % 360

        geometric_information = pd.DataFrame(
            {
                "cell_idx1": self._connected_cells["cell_idx1"].values,
                "cell_idx2": self._connected_cells["cell_idx2"].values,
                "cl1": cl1,
                "cl2": cl2,
                "hwva": hwva,
                "angldegx": angledegx,
                "cdist": cdist,
            }
        )

        return geometric_information

    @classmethod
    def _create_global_to_local_idx(
        cls, partition_info: List[PartitionInfo], global_cell_indices: GridDataArray
    ) -> Dict[int, pd.DataFrame]:
        global_to_local_idx = {}
        for submodel_partition_info in partition_info:
            local_cell_indices = cls._get_local_cell_indices(submodel_partition_info)

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
