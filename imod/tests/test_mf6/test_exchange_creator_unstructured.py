import numpy as np
import pytest
import xugrid as xu

import imod
from imod.mf6.exchange_creator_unstructured import ExchangeCreator_Unstructured
from imod.mf6.modelsplitter import create_partition_info
from imod.tests.fixtures.flow_basic_fixture import BasicDisSettings
from imod.typing.grid import zeros_like


def _create_submodel_labels_unstructured(idomain, number_partitions):
    submodel_labels = zeros_like(idomain.sel(layer=1))
    dimension = len(submodel_labels)
    switch_index = int(dimension / number_partitions)

    submodel_labels[:] = number_partitions - 1
    for ipartition in range(number_partitions):
        start = ipartition * switch_index
        end = (ipartition + 1) * switch_index
        submodel_labels[start:end] = ipartition

    return submodel_labels


def to_unstruct_domain(idomain):
    grid = xu.Ugrid2d.from_structured(idomain)

    domain_data = imod.util.ugrid2d_data(idomain, grid.face_dimension)
    return xu.UgridDataArray(domain_data, grid)


class TestExchangeCreator_Unstructured:
    @pytest.mark.parametrize("number_partitions", [2, 3, 5])
    def test_create_exchanges_unstructured_validate_number_of_exchanges(
        self,
        unstructured_flow_simulation: imod.mf6.Modflow6Simulation,
        number_partitions: int,
    ):
        # Arrange.
        idomain = unstructured_flow_simulation["flow"].domain
        submodel_labels = _create_submodel_labels_unstructured(
            idomain, number_partitions
        )
        partition_info = create_partition_info(submodel_labels)
        exchange_creator = ExchangeCreator_Unstructured(submodel_labels, partition_info)

        # Act.
        exchanges = exchange_creator.create_exchanges("flow", idomain.layer)

        # Assert.
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

        # Arrange.
        number_partitions = 2
        idomain = unstructured_flow_simulation["flow"].domain
        submodel_labels = _create_submodel_labels_unstructured(
            idomain, number_partitions
        )
        partition_info = create_partition_info(submodel_labels)
        exchange_creator = ExchangeCreator_Unstructured(submodel_labels, partition_info)

        # Act.
        exchanges = exchange_creator.create_exchanges("flow", idomain.layer)

        # Assert.
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

    @pytest.mark.parametrize(
        "parameterizable_basic_dis",
        [BasicDisSettings(nlay=1, nrow=4, ncol=4, space_generator=np.geomspace)],
        indirect=True,
    )
    def test_create_exchanges_unstructured_validate_geometric_coefficients(
        self, parameterizable_basic_dis
    ):
        # Arrange.
        number_partitions = 2

        idomain, _, _ = parameterizable_basic_dis
        expected_cl1 = abs(idomain.dy / 2)[1].values
        expected_cl2 = abs(idomain.dy / 2)[2].values
        expected_hwva = idomain.dx.values

        idomain = to_unstruct_domain(idomain)

        submodel_labels = _create_submodel_labels_unstructured(
            idomain, number_partitions
        )
        partition_info = create_partition_info(submodel_labels)
        exchange_creator = ExchangeCreator_Unstructured(submodel_labels, partition_info)

        # Act.
        exchanges = exchange_creator.create_exchanges("flow", idomain.layer)

        # Assert.
        assert np.allclose(exchanges[0].dataset["cl1"], expected_cl1)
        assert np.allclose(exchanges[0].dataset["cl2"], expected_cl2)
        assert np.allclose(exchanges[0].dataset["hwva"], expected_hwva)

    @pytest.mark.parametrize(
        "parameterizable_basic_dis",
        [BasicDisSettings(nlay=1, nrow=4, ncol=4, space_generator=np.linspace)],
        indirect=True,
    )
    @pytest.mark.parametrize("partition_axis", ["x", "y"])
    def test_create_exchanges_unstructured_validate_auxiliary_coefficients(
        self, parameterizable_basic_dis, partition_axis
    ):
        # Arrange.
        idomain, _, _ = parameterizable_basic_dis
        expected_angledegx = 0.0 if partition_axis == "x" else 90.0
        expected_cdist = abs(
            idomain.coords[partition_axis][1] - idomain.coords[partition_axis][2]
        ).values

        idomain = to_unstruct_domain(idomain)

        submodel_labels = zeros_like(idomain.sel(layer=1))
        if partition_axis == "x":
            submodel_labels = submodel_labels.where(
                submodel_labels.grid.face_x < 0.5, 1, 0
            )
        else:
            submodel_labels = submodel_labels.where(
                submodel_labels.grid.face_y < 0.5, 1, 0
            )

        partition_info = create_partition_info(submodel_labels)
        exchange_creator = ExchangeCreator_Unstructured(submodel_labels, partition_info)

        # Act.
        exchanges = exchange_creator.create_exchanges("flow", idomain.layer)

        # Assert.
        assert np.allclose(
            exchanges[0].dataset["auxiliary"].sel(variable="angledegx"),
            expected_angledegx,
        )
        assert np.allclose(
            exchanges[0].dataset["auxiliary"].sel(variable="cdist"), expected_cdist
        )
