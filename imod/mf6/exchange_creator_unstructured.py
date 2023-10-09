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

    @property
    def _coordinate_names(self):
        return ["cell_index"]

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
