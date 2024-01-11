from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.multimodel.exchange_creator import ExchangeCreator
from imod.mf6.multimodel.modelsplitter import PartitionInfo
from imod.typing import GridDataArray


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
        self._connected_cell_edge_indices = (
            self._find_subdomain_connection_edge_indices(submodel_labels)
        )
        super().__init__(submodel_labels, partition_info)

    @property
    def _coordinate_names(self):
        return ["cell_index"]

    def _find_connected_cells(self) -> pd.DataFrame:
        edge_face_connectivity = self._submodel_labels.ugrid.grid.edge_face_connectivity

        face1 = edge_face_connectivity[self._connected_cell_edge_indices, 0]
        face2 = edge_face_connectivity[self._connected_cell_edge_indices, 1]

        label_of_face1 = self._submodel_labels.values[face1]
        label_of_face2 = self._submodel_labels.values[face2]

        connected_cell_info = pd.DataFrame(
            {
                "cell_idx1": face1,
                "cell_idx2": face2,
                "cell_label1": label_of_face1,
                "cell_label2": label_of_face2,
            }
        )

        return connected_cell_info

    def _compute_geometric_information(self) -> pd.DataFrame:
        grid = self._submodel_labels.ugrid.grid
        face1, face2 = self._get_partition_sorted_connected_faces()
        centroid_1 = grid.centroids[face1]
        centroid_2 = grid.centroids[face2]

        cdist = np.linalg.norm(centroid_2 - centroid_1, axis=1)

        edge_coordinates = grid.edge_node_coordinates[self._connected_cell_edge_indices]

        U = np.diff(edge_coordinates, axis=1)[:, 0]
        # Compute vector of first cell centroid to first edge vertex
        Vi = centroid_1 - edge_coordinates[:, 0]
        # Compute vector of second cell centroid to first edge vertex
        Vj = centroid_2 - edge_coordinates[:, 0]
        length = np.linalg.norm(U, axis=1)

        # get the normal to the cell edge from U
        dx = U[:, 0]
        dy = U[:, 1]
        normal = np.array((dy[:], -dx[:]), dtype=np.float_).T
        
        # If the inner product of the normal with a vector on the edge to the face centroid is positive
        # then the normal vector points inwards
        inward_vector_mask = np.sum(normal * Vi, axis=-1) > 0
        normal[inward_vector_mask] = -normal[inward_vector_mask]

        angle = np.degrees(np.arctan2(normal[:, 1], normal[:, 0]))

        df = pd.DataFrame(
            {
                "cell_idx1": self._connected_cells["cell_idx1"].values,
                "cell_idx2": self._connected_cells["cell_idx2"].values,
                "cl1": np.abs(np.cross(U, Vi)) / length,
                "cl2": np.abs(np.cross(U, Vj)) / length,
                "hwva": length,
                "angldegx": angle,
                "cdist": cdist,
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

    @staticmethod
    def _find_subdomain_connection_edge_indices(submodel_labels):
        edge_face_connectivity = submodel_labels.ugrid.grid.edge_face_connectivity

        face1 = edge_face_connectivity[:, 0]
        face2 = edge_face_connectivity[:, 1]

        label_of_face1 = submodel_labels.values[face1]
        label_of_face2 = submodel_labels.values[face2]

        is_internal_edge = label_of_face1 - label_of_face2 == 0
        is_external_boundary_edge = np.any((face1 == -1, face2 == -1), axis=0)
        is_inter_domain_edge = ~is_internal_edge & ~is_external_boundary_edge

        return is_inter_domain_edge

    def _get_partition_numbers(self, face: np.ndarray) -> np.ndarray:
        return self._submodel_labels.data[face]

    def _get_partition_sorted_connected_faces(self) -> (np.ndarray, np.ndarray):
        grid = self._submodel_labels.ugrid.grid
        edge_face_connectivity = grid.edge_face_connectivity

        unordered_face1, unordered_face2 = edge_face_connectivity[
            self._connected_cell_edge_indices
        ].T

        # Obtain the cellface indices on both sides of each edge.
        # They should be ordered by giving the cellface with the lowest partition number first.
        face_partition_1 = self._get_partition_numbers(unordered_face1)
        face_partition_2 = self._get_partition_numbers(unordered_face2)
        face1 = np.where(
            face_partition_1 > face_partition_2, unordered_face2, unordered_face1
        )
        face2 = np.where(
            face_partition_1 > face_partition_2, unordered_face1, unordered_face2
        )
        return (face1, face2)
