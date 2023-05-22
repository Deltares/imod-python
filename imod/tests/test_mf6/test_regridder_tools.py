from imod.mf6.regridding_tools import RegridderInstancesCollection
def test_instance_collection_returns_same_instance_when_regridder_and_method_match(basic_unstructured_dis):
    idomain, top, bot = basic_unstructured_dis
    collection = RegridderInstancesCollection()


