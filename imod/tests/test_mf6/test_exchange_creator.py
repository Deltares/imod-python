from collections import namedtuple

import numpy as np
import pytest
import xarray as xr
from pytest_cases import parametrize_with_cases
import imod
from imod.mf6.exchange_creator import ExchangeCreator
from imod.mf6.modelsplitter import create_partition_info
from imod.typing.grid import zeros_like

ExpectedExchanges = namedtuple("ExpectedExchanges", "cell_id1 cell_id2")


def _create_submodel_labels(active, x_number_partitions, y_number_partitions):
    x_split_location = np.linspace(
        active.x.min(), active.x.max(), x_number_partitions + 1
    )
    y_split_location = np.linspace(
        active.y.min(), active.y.max(), y_number_partitions + 1
    )

    coords = active.coords

    submodel_labels = zeros_like(active.sel(layer=1))
    for id_x in np.arange(0, x_number_partitions):
        for id_y in np.arange(0, y_number_partitions):
            label_id = np.ravel_multi_index(
                [id_x, id_y], (x_number_partitions, y_number_partitions)
            )
            submodel_labels.loc[
                (coords["y"] >= y_split_location[id_y])
                & (coords["y"] <= y_split_location[id_y + 1]),
                (coords["x"] >= x_split_location[id_x])
                & (coords["x"] <= x_split_location[id_x + 1]),
            ] = label_id

    return submodel_labels

def create_submodel_labels_unstructured(idomain, number_partitions):
    
    submodel_labels = zeros_like(idomain.sel(layer=1))
    dimension = len(submodel_labels)
    switch_index = int(dimension / number_partitions)

    submodel_labels[:] = number_partitions -1
    for ipartition in range(number_partitions):
        start = ipartition*switch_index
        end = (ipartition+1) * switch_index
        submodel_labels[start: end] = ipartition

    return submodel_labels




class TestExchangeCreator:
    @pytest.mark.parametrize(
        "x_number_partitions, y_number_partitions", [(1, 1), (3, 1), (1, 3), (3, 3)]
    )
    @pytest.mark.parametrize("parameterizable_basic_dis", [(1, 3, 3)], indirect=True)
    def test_create_exchanges_validate_number_of_exchanges(
        self, x_number_partitions, y_number_partitions, parameterizable_basic_dis
    ):
        # Arrange.
        idomain, _, _ = parameterizable_basic_dis
        submodel_labels = _create_submodel_labels(
            idomain, x_number_partitions, y_number_partitions
        )
        partition_info = create_partition_info(submodel_labels)

        exchange_creator = ExchangeCreator(submodel_labels, partition_info)

        model_name = "test_model"
        layer = idomain.layer

        # Act.
        exchanges = exchange_creator.create_exchanges(model_name, layer)

        # Assert.
        num_exchanges_x_direction = y_number_partitions * (x_number_partitions - 1)
        num_exchanges_y_direction = x_number_partitions * (y_number_partitions - 1)
        assert len(exchanges) == num_exchanges_x_direction + num_exchanges_y_direction

    @pytest.mark.parametrize("number_partitions", [2 ,3, 5])
    def test_find_connected_cells_unstructured( self, unstructured_flow_simulation: imod.mf6.Modflow6Simulation, number_partitions: int):
        idomain = unstructured_flow_simulation["flow"]["disv"]["idomain"]
        submodel_labels = create_submodel_labels_unstructured(idomain, number_partitions)
        partition_info = create_partition_info(submodel_labels)
        exchange_creator = ExchangeCreator(submodel_labels, partition_info)

        exchange_creator._find_connected_cells_unstructured()

        # Act.
        exchanges = exchange_creator.create_exchanges("flow", idomain.layer)

        # assert
        assert len(exchanges) == number_partitions -1


    class ExpectedCellIds:
        @staticmethod
        def case_split_along_x_axis():
            """
            submodel_labels
            [[1, 1, 1, 1],
             [1, 1, 1, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]])
            """
            x_number_partitions = 1
            y_number_partitions = 2

            # test_model_0 <-> test_model_1
            expected_cell_id1 = np.array([(2, 1), (2, 2), (2, 3), (2, 4)])
            expected_cell_id2 = np.array([(1, 1), (1, 2), (1, 3), (1, 4)])

            return (
                x_number_partitions,
                y_number_partitions,
                [ExpectedExchanges(expected_cell_id1, expected_cell_id2)],
            )

        @staticmethod
        def case_split_along_y_axis():
            """
            submodel_labels
            [[0, 0, 1, 1],
             [0, 0, 1, 1],
             [0, 0, 1, 1],
             [0, 0, 1, 1]])
            """
            x_number_partitions = 2
            y_number_partitions = 1

            # test_model_0 <-> test_model_1
            expected_cell_id1 = np.array([(1, 2), (2, 2), (3, 2), (4, 2)])
            expected_cell_id2 = np.array([(1, 1), (2, 1), (3, 1), (4, 1)])

            return (
                x_number_partitions,
                y_number_partitions,
                [ExpectedExchanges(expected_cell_id1, expected_cell_id2)],
            )

        @staticmethod
        def case_split_along_both_axis():
            """
            submodel_labels
            [[1, 1, 3, 3],
             [1, 1, 3, 3],
             [0, 0, 2, 2],
             [0, 0, 2, 2]]
            """
            x_number_partitions = 2
            y_number_partitions = 2

            # test_model_0 <-> test_model_2
            exchange1_expected_cell_id1 = np.array([(1, 2), (2, 2)])
            exchange1_expected_cell_id2 = np.array([(1, 1), (2, 1)])

            # test_model_1 <-> test_model_0
            exchange2_expected_cell_id1 = np.array([(2, 1), (2, 2)])
            exchange2_expected_cell_id2 = np.array([(1, 1), (1, 2)])

            # test_model_1 <-> test_model_3
            exchange3_expected_cell_id1 = np.array([(1, 2), (2, 2)])
            exchange3_expected_cell_id2 = np.array([(1, 1), (2, 1)])

            # test_model_1 <-> test_model_2
            exchange4_expected_cell_id1 = np.array([(2, 1), (2, 2)])
            exchange4_expected_cell_id2 = np.array([(1, 1), (1, 2)])

            return (
                x_number_partitions,
                y_number_partitions,
                [
                    ExpectedExchanges(
                        exchange1_expected_cell_id1, exchange1_expected_cell_id2
                    ),
                    ExpectedExchanges(
                        exchange2_expected_cell_id1, exchange2_expected_cell_id2
                    ),
                    ExpectedExchanges(
                        exchange3_expected_cell_id1, exchange3_expected_cell_id2
                    ),
                    ExpectedExchanges(
                        exchange4_expected_cell_id1, exchange4_expected_cell_id2
                    ),
                ],
            )

    @parametrize_with_cases(
        "x_number_partitions,y_number_partitions,expected_exchanges",
        cases=ExpectedCellIds,
    )
    @pytest.mark.parametrize("parameterizable_basic_dis", [(1, 4, 4)], indirect=True)
    def test_create_exchanges_validate_local_cell_ids(
        self,
        parameterizable_basic_dis,
        x_number_partitions,
        y_number_partitions,
        expected_exchanges,
    ):
        # Arrange.
        idomain, _, _ = parameterizable_basic_dis
        submodel_labels = _create_submodel_labels(
            idomain, x_number_partitions, y_number_partitions
        )
        partition_info = create_partition_info(submodel_labels)

        exchange_creator = ExchangeCreator(submodel_labels, partition_info)

        model_name = "test_model"
        layer = idomain.layer

        # Act.
        exchanges = exchange_creator.create_exchanges(model_name, layer)

        # Assert.
        assert len(exchanges) == len(expected_exchanges)
        for idx in range(len(exchanges)):
            np.testing.assert_array_equal(
                exchanges[idx].dataset["cell_id1"].values, expected_exchanges[idx][0]
            )
            np.testing.assert_array_equal(
                exchanges[idx].dataset["cell_id2"].values, expected_exchanges[idx][1]
            )
