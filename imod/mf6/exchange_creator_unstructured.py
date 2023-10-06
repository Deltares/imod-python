import copy
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.exchange_creator import ExchangeCreator
from imod.mf6.modelsplitter import PartitionInfo
from imod.typing.grid import GridDataArray


class ExchangeCreator_Unstructured(ExchangeCreator):
    """
    Creates the GroundWaterFlow to GroundWaterFlow exchange package (gwfgwf) as a function of a submodel label array and a
    PartitionInfo object. This file contains the cell indices of coupled cells. With coupled cells we mean cells that are adjacent but
    that are located in different subdomains.  At the moment only structured grids are supported, for unstructured grids the geometric information
    is still set to default values.

    The submodel_labels array should have the same topology as the domain being partitioned. The array will be used
    to determine the connectivity of the submodels after the split operation has been performed.

    """

    def __init__(
        self, submodel_labels: GridDataArray, partition_info: List[PartitionInfo]
    ):
        super().__init__(submodel_labels, partition_info)

    @classmethod
    def _to_xarray(cls, connected_cells: pd.DataFrame) -> xr.Dataset:
        dataset = connected_cells.to_xarray()

        size = connected_cells.shape[0]
        cell_id1 = np.array(connected_cells["cell_id1"])
        cell_id1_2d = cell_id1.reshape(size, 1)

        dataset["cell_id1"] = xr.DataArray(
            cell_id1_2d,
            dims=("index", "cell_dims1"),
            coords={"cell_dims1": ["cellindex1"]},
        )

        cell_id2 = np.array(connected_cells["cell_id2"])
        cell_id2_2d = cell_id2.reshape(size, 1)
        dataset["cell_id2"] = xr.DataArray(
            cell_id2_2d,
            dims=("index", "cell_dims2"),
            coords={"cell_dims2": ["cellindex2"]},
        )
        return dataset

    def _adjust_gridblock_indexing(
        self, connected_cells_dataset: xr.Dataset
    ) -> xr.Dataset:
        connected_cells_dataset["cell_id1"].values = (
            connected_cells_dataset["cell_id1"].values + 1
        )
        connected_cells_dataset["cell_id2"].values = (
            connected_cells_dataset["cell_id2"].values + 1
        )
        return connected_cells_dataset

    def _find_connected_cells(self) -> pd.DataFrame:
        # make a deepcopy to avoid changing the original
        edge_face = copy.deepcopy(
            self._submodel_labels.ugrid.grid.edge_face_connectivity
        )
        edge_index = np.arange(len(edge_face))

        f1 = edge_face[:, 0]
        f2 = edge_face[:, 1]

        # edges at the external boundary have one -1 for the external "gridblock"
        # we set both entries to -1 here so that en exteral edge will have [-1, -1]
        f1 = np.where(f2 >= 0, f1, -1)
        f2 = np.where(f1 >= 0, f2, -1)
        label_of_edge1 = self._submodel_labels.values[f1]
        label_of_edge2 = self._submodel_labels.values[f2]

        # only keep the edge indces where the labels are different. The others will be -1
        edge_indices_internal_boundary = np.where(
            label_of_edge1 - label_of_edge2 != 0, edge_index, -1
        )
        # only keep the edge indces hat are not -1
        edge_indices_internal_boundary = np.setdiff1d(
            edge_indices_internal_boundary, [-1]
        )
        internal_boundary = edge_face[edge_indices_internal_boundary]

        connected_cell_info = pd.DataFrame(
            {
                "cell_idx1": internal_boundary[:, 0],
                "cell_idx2": internal_boundary[:, 1],
                "cell_label1": label_of_edge1[edge_indices_internal_boundary],
                "cell_label2": label_of_edge2[edge_indices_internal_boundary],
            }
        )

        return connected_cell_info

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
        ones = np.ones_like(self._connected_cells["cell_idx1"].values, dtype=int)

        df = pd.DataFrame(
            {
                "cell_idx1": self._connected_cells["cell_idx1"].values,
                "cell_idx2": self._connected_cells["cell_idx2"].values,
                "cl1": ones,
                "cl2": ones,
                "hwva": ones,
            }
        )
        return df
