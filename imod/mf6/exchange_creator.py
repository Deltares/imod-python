import abc
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.gwfgwf import GWFGWF
from imod.mf6.modelsplitter import PartitionInfo
from imod.mf6.utilities.grid_utilities import get_active_domain_slice, to_cell_idx
from imod.typing.grid import GridDataArray, is_unstructured


class ExchangeCreator(abc.ABC):
    """
    Creates the GroundWaterFlow to GroundWaterFlow exchange package (gwfgwf) as a function of a submodel label array and a
    PartitionInfo object. This file contains the cell indices of coupled cells. With coupled cells we mean cells that are adjacent but
    that are located in different subdomains.  At the moment only structured grids are supported, for unstructured grids the geometric information
    is still set to default values.

    The submodel_labels array should have the same topology as the domain being partitioned. The array will be used
    to determine the connectivity of the submodels after the split operation has been performed.

    """

    @classmethod
    def _to_xarray(cls, connected_cells: pd.DataFrame) -> xr.Dataset:
        raise NotImplementedError

    def _find_connected_cells(self) -> pd.DataFrame:
        """
        abstract method that creates a dataframe with the cell indices and subdomain indices of cells that are on the
        boundary of subdomains. The cell index is a number assigned to each cell on the unpartitioned domain
        For structured grids it replaces the row and column index which are usually used to identify a cell.
        """
        raise NotImplementedError

    def _adjust_gridblock_indexing(self, connected_cells: xr.Dataset) -> xr.Dataset:
        """
        abstract method that adjusts the gridblock numbering from 0-based to 1-based.
        """
        return connected_cells

    def _compute_geometric_information(self) -> pd.DataFrame:
        """
        abstract method that computes the geometric information needed for the gwfgwf files such as
        ihc, cl1, cl2, hwva ( see modflow documentation )
        """
        raise NotImplementedError

    def _create_global_to_local_idx(
        partition_info: List[PartitionInfo], global_cell_indices: GridDataArray
    ) -> Dict[int, pd.DataFrame]:
        """
        abstract method that creates for each partition a maping from global cell indices to local cells in that partition.
        """
        raise NotImplementedError

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
        layers = layers.to_dataframe().filter(["layer"])

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

                connected_cells_dataset = self._to_xarray(connected_cells)

                self._adjust_gridblock_indexing(connected_cells_dataset)

                exchanges.append(
                    GWFGWF(
                        f"{model_name}_{model_id1}",
                        f"{model_name}_{model_id2}",
                        **connected_cells_dataset,
                    )
                )

        return exchanges

    def _create_global_cellidx_to_local_cellid_mapping(
        self, partition_info: List[PartitionInfo]
    ) -> Dict[int, pd.DataFrame]:
        global_to_local_idx = self._create_global_to_local_idx(
            partition_info, self._global_cell_indices
        )
        local_cell_idx_to_id = self._local_cell_idx_to_id(partition_info)

        mapping = {}
        for submodel_partition_info in partition_info:
            model_id = submodel_partition_info.id
            mapping[model_id] = pd.merge(
                global_to_local_idx[model_id], local_cell_idx_to_id[model_id]
            )

        return mapping

    @classmethod
    def _get_local_cell_indices(
        cls, submodel_partition_info: PartitionInfo
    ) -> xr.DataArray:
        domain_slice = get_active_domain_slice(submodel_partition_info.active_domain)
        local_domain = submodel_partition_info.active_domain.sel(domain_slice)

        return to_cell_idx(local_domain)

    @classmethod
    def _local_cell_idx_to_id(cls, partition_info) -> Dict[int, pd.DataFrame]:
        local_cell_idx_to_id = {}
        for submodel_partition_info in partition_info:
            local_cell_indices = cls._get_local_cell_indices(submodel_partition_info)

            if is_unstructured(local_cell_indices):
                model_id = submodel_partition_info.id
                local_cell_idx_to_id[model_id] = pd.DataFrame(
                    {
                        "local_idx": local_cell_indices,
                        "local_cell_id": local_cell_indices,
                    }
                )
            else:
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
