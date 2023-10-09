import copy
from typing import Dict, List

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
        """
        converts a panda dataframe with exchange data to an xarray dataset. The
        dataframe must have columns called cell_id1 and cell_id2: indices of
        cells that are part of the exchange boundary (the subdomain boundary, on
        both sides of the boundary)
        """
        dataset = connected_cells.to_xarray()

        size = connected_cells.shape[0]
        cell_id1 = np.array(connected_cells["cell_id1"])

        dataset["cell_id1"] = xr.DataArray(
            cell_id1.reshape(size, 1),
            dims=("index", "cell_dims1"),
            coords={"cell_dims1": ["cellindex1"]},
        )

        cell_id2 = np.array(connected_cells["cell_id2"])

        dataset["cell_id2"] = xr.DataArray(
            cell_id2.reshape(size, 1),
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
        edge_face_connectivity = self._submodel_labels.ugrid.grid.edge_face_connectivity

        face1 = edge_face_connectivity[:, 0]
        face2 = edge_face_connectivity[:, 1]

        label_of_face1 = self._submodel_labels.values[face1]
        label_of_face2 = self._submodel_labels.values[face2]

        is_internal_edge = label_of_face1 - label_of_face2 == 0
        is_external_boundary_edge = np.any((face1 == -1, face2 == -1), axis=0)
        is_inter_domain_edge = ~is_internal_edge & ~is_external_boundary_edge

        connected_cell_info = pd.DataFrame(
            {
                "cell_idx1": face1[is_inter_domain_edge],
                "cell_idx2": face2[is_inter_domain_edge],
                "cell_label1": label_of_face1[is_inter_domain_edge],
                "cell_label2": label_of_face2[is_inter_domain_edge],
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

    @classmethod
    def _create_global_to_local_idx(
        cls, partition_info: List[PartitionInfo], global_cell_indices: GridDataArray
    ) -> Dict[int, pd.DataFrame]:
        global_to_local_idx = {}
        for submodel_partition_info in partition_info:
            local_cell_indices = cls._get_local_cell_indices(submodel_partition_info)

            global_cell_indices_partition = global_cell_indices.where(
                submodel_partition_info.active_domain == 1
            )
            global_cell_indices_partition = global_cell_indices_partition.dropna(
                "mesh2d_nFaces", how="all"
            )

            global_cell_indices_df = global_cell_indices_partition.to_dataframe()
            global_cell_indices_da = xr.Dataset.from_dataframe(global_cell_indices_df)

            overlap = xr.merge(
                (global_cell_indices_da, xr.DataArray(local_cell_indices)),
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
