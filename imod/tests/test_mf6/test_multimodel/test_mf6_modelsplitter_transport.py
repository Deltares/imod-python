import pytest
from imod.typing.grid import zeros_like
from imod.mf6.multimodel.modelsplitter import create_partition_info, slice_model

def test_slice_model_structured(flow_transport_simulation):

    transport_model = flow_transport_simulation["tpt_a"]
    submodel_labels = zeros_like(transport_model.domain)
    submodel_labels[:,:,30:] = 1

    partition_info = create_partition_info(submodel_labels)

    submodel_list = []
    for submodel_partition_info in partition_info:
        new_model_name = f"transport_{submodel_partition_info.id}"
        submodel_list.append( slice_model(
            submodel_partition_info, transport_model
        ))