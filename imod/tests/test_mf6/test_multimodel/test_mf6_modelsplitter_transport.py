from imod.mf6.multimodel.modelsplitter import create_partition_info, slice_model
from imod.typing.grid import zeros_like


def test_slice_model_structured(flow_transport_simulation):
    # Arrange.
    transport_model = flow_transport_simulation["tpt_a"]
    submodel_labels = zeros_like(transport_model.domain)
    submodel_labels[:, :, 30:] = 1

    partition_info = create_partition_info(submodel_labels)

    submodel_list = []

    # Act
    for submodel_partition_info in partition_info:
        submodel_list.append(slice_model(submodel_partition_info, transport_model))

    # Assert
    assert len(submodel_list) == 2
    for submodel in submodel_list:
        for package_name in list(transport_model.keys()):
            assert package_name in list(submodel.keys())
