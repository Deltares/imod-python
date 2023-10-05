import numpy as np
import pytest

import imod
from imod.mf6.exchange_creator_unstructured import ExchangeCreator_Unstructured
from imod.mf6.modelsplitter import create_partition_info
from imod.typing.grid import zeros_like


def create_submodel_labels_unstructured(idomain, number_partitions):
    submodel_labels = zeros_like(idomain.sel(layer=1))
    dimension = len(submodel_labels)
    switch_index = int(dimension / number_partitions)

    submodel_labels[:] = number_partitions - 1
    for ipartition in range(number_partitions):
        start = ipartition * switch_index
        end = (ipartition + 1) * switch_index
        submodel_labels[start:end] = ipartition

    return submodel_labels


class TestExchangeCreator_Unstructured:
    @pytest.mark.parametrize("number_partitions", [2, 3, 5])
    def test_create_exchanges_unstructured_validate_number_of_exchanges(
        self,
        unstructured_flow_simulation: imod.mf6.Modflow6Simulation,
        number_partitions: int,
    ):
        idomain = unstructured_flow_simulation["flow"]["disv"]["idomain"]
        submodel_labels = create_submodel_labels_unstructured(
            idomain, number_partitions
        )
        partition_info = create_partition_info(submodel_labels)
        exchange_creator = ExchangeCreator_Unstructured(submodel_labels, partition_info)

        exchange_creator._find_connected_cells()

        # Act.
        exchanges = exchange_creator.create_exchanges("flow", idomain.layer)

        # assert
        assert len(exchanges) == number_partitions - 1

    def test_create_exchanges_unstructured_validate_exchange_locations(
        self,
        unstructured_flow_simulation: imod.mf6.Modflow6Simulation,
    ):
        # The test domain is regular with dimensions 6 (row) x 6 (column) x3
        # (layer) but is saved as an unstructured grid. We split it in 2 on with
        # the first 3 rows being domain 1 and the last 3 rows being domain 2.
        # Hence the exchanges should be located on cells 13, 14, 15, 16 , 17 and
        # 18 ( the third row). and these indices should occur for every layer of
        # the 3. On the second domain, the exchanges should be located on cells
        # 1 to 6
        number_partitions = 2
        idomain = unstructured_flow_simulation["flow"]["disv"]["idomain"]
        submodel_labels = create_submodel_labels_unstructured(
            idomain, number_partitions
        )
        partition_info = create_partition_info(submodel_labels)
        exchange_creator = ExchangeCreator_Unstructured(submodel_labels, partition_info)

        exchange_creator._find_connected_cells()

        # Act.
        exchanges = exchange_creator.create_exchanges("flow", idomain.layer)

        # assert
        nlayer = 3
        cell_id1, counts = np.unique(
            exchanges[0].dataset["cell_id1"].values, return_counts=True
        )
        cell_id1_dict = dict(zip(cell_id1, counts))
        for icell in range(13, 19, 1):
            assert cell_id1_dict[icell] == nlayer

        cell_id2, counts = np.unique(
            exchanges[0].dataset["cell_id2"].values, return_counts=True
        )
        cell_id2_dict = dict(zip(cell_id2, counts))
        for icell in range(1, 6, 1):
            assert cell_id2_dict[icell] == nlayer
