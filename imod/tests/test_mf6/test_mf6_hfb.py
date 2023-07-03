import imod
import geopandas as gpd
import numpy as np
import shapely
import xugrid as xu
import pytest

@pytest.mark.parametrize("hfb_specialization",[imod.mf6.HorizontalFlowBarrierResistance, imod.mf6.HorizontalFlowBarrierMultiplier, imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic] )
def test_hfb_writing_1layer(hfb_specialization, tmp_path, unstructured_flow_model):
    
    idomain= unstructured_flow_model["disv"]["idomain"]
    # x = [9.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0]
    #y = x[::-1]
    x = [0. ,2. ,2., 3., 3., 5.]
    y = [5., 5. ,5., 5., 5., 5.]
    indices = np.repeat(np.arange(3), 2)
    linestrings = shapely.linestrings(x, y, indices=indices)
    lines = gpd.GeoDataFrame(geometry=linestrings)
    lines["linedata"] = 10.0

    ugrid_1d = xu.UgridDataset.from_geodataframe(lines)

    line_as_dataarray =ugrid_1d["linedata"]
    line_as_dataarray=line_as_dataarray.expand_dims("layer")
    line_as_dataarray=line_as_dataarray.assign_coords(layer=[1])

    hfb = hfb_specialization(line_as_dataarray, idomain )
    hfb.render(tmp_path, "hfb", None, False)
    hfb.write(tmp_path, "hfb", None, False)
 