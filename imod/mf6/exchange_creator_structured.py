from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.exchange_creator import ExchangeCreator
from imod.mf6.modelsplitter import PartitionInfo
from imod.mf6.utilities.grid_utilities import create_geometric_grid_info
from imod.typing.grid import GridDataArray


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

    @classmethod
    def _to_xarray(cls, connected_cells: pd.DataFrame) -> xr.Dataset:
        """
        converts a panda dataframe with exchange data to an xarray dataset. The
        dataframe must have columns called cell_id1, row_1, column_1, cell_id2,
        row_2 and col_2 containing the cell_id, row and column indices of cells
        that are part of the exchange boundary (the subdomain boundary, on both
        sides of the boundary)
        """
        dataset = connected_cells.to_xarray()

        dataset["cell_id1"] = xr.DataArray(
            np.array(list(zip(*connected_cells["cell_id1"]))).T,
            dims=("index", "cell_dims1"),
            coords={"cell_dims1": ["row_1", "column_1"]},
        )
        dataset["cell_id2"] = xr.DataArray(
            np.array(list(zip(*connected_cells["cell_id2"]))).T,
            dims=("index", "cell_dims2"),
            coords={"cell_dims2": ["row_2", "column_2"]},
        )

        return dataset

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
