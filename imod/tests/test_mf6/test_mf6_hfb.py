import imod
import geopandas as gpd
import numpy as np
import shapely
import xugrid as xu

def test_hfb_resistance(tmp_path, circle_model):
    
    ibound= circle_model["GWF_1"]["disv"]["idomain"]
    x = [10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 50.0]
    y = x[::-1]

    indices = np.repeat(np.arange(4), 2)
    linestrings = shapely.linestrings(x, y, indices=indices)
    lines = gpd.GeoDataFrame(geometry=linestrings)
    lines["resistance"] = 10.0

    ugrid_1d = xu.UgridDataset.from_geodataframe(lines)

    c =ugrid_1d["resistance"]
    c=c.expand_dims("layer")
    c=c.assign_coords(layer=[1])

    hfb = imod.mf6.HorizontalFlowBarrierResistance(c, ibound )
    hfb.render(tmp_path, "hfb", None, False)
    hfb.write(tmp_path, "hfb", None, False)
    
def test_hfb_multiplier(tmp_path, circle_model):
    
    idomain= circle_model["GWF_1"]["disv"]["idomain"]
    x = [10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 50.0]
    y = x[::-1]

    indices = np.repeat(np.arange(4), 2)
    linestrings = shapely.linestrings(x, y, indices=indices)
    lines = gpd.GeoDataFrame(geometry=linestrings)
    lines["resistance"] = 10.0

    ugrid_1d = xu.UgridDataset.from_geodataframe(lines)

    c =ugrid_1d["resistance"]
    c=c.expand_dims("layer")
    c=c.assign_coords(layer=[1])

    hfb = imod.mf6.HorizontalFlowBarrierMultiplier(c, idomain )
    hfb.render(tmp_path, "hfb", None, False)
    hfb.write(tmp_path, "hfb", None, False)
    
def test_hfb_hydraulic_characteristic(tmp_path, circle_model):

    idomain= circle_model["GWF_1"]["disv"]["idomain"]
    x = [10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 50.0]
    y = x[::-1]

    indices = np.repeat(np.arange(4), 2)
    linestrings = shapely.linestrings(x, y, indices=indices)
    lines = gpd.GeoDataFrame(geometry=linestrings)
    lines["resistance"] = 10.0

    ugrid_1d = xu.UgridDataset.from_geodataframe(lines)

    c =ugrid_1d["resistance"]
    c=c.expand_dims("layer")
    c=c.assign_coords(layer=[1])

    hfb = imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic(c, idomain )
    hfb.render(tmp_path, "hfb", None, False)
    hfb.write(tmp_path, "hfb", None, False)
    
