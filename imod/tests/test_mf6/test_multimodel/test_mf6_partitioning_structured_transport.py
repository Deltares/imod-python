from imod.mf6.multimodel.modelsplitter import create_partition_info, slice_model
from imod.typing.grid import zeros_like


def test_partition_structured(flow_transport_simulation):
    # Arrange.
    transport_model = flow_transport_simulation["tpt_a"]
    submodel_labels = zeros_like(transport_model.domain.isel(layer=0))
    submodel_labels[:, 30:] = 1

    split_simulation = flow_transport_simulation.split(submodel_labels)