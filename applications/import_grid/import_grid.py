
from os import listdir
from os.path import isfile, join
import re
import numpy as np

import xarray
from imod import idf, mf6
import re
from pathlib import Path

relative_paths = { "BOT": "BOT\\VERSION_1\\",
 "TOP":"TOP\\VERSION_1\\" ,
 "KD":"\KDW\VERSION_1",
  }

def get_idf_filepaths(dir, basename):
    return  [Path(dir).joinpath(f) for f in listdir(dir) if isfile(join(dir, f)) and f.startswith(basename) and f.endswith(".IDF")]

def get_layer_number_from_filenpath(filepath):
    filename = filepath.name
    numbers = re.findall(r'\d+', filename)
    if len(numbers)==1:
        return int(numbers[0])
    assert False, "could not find 1 number in filename"


def convert_to_netcdf_per_layer(idf_files, data_name):
    per_layer ={}
    for f in idf_files:
        layer = get_layer_number_from_filenpath(f)
        netcdf =idf.open(f)
        per_layer[layer]= netcdf

    nrlayers = len(per_layer)
    result = per_layer[1]
    result =result.expand_dims({"layer": [1]})
    result.name = data_name
    for ilayer in range(2, nrlayers):
        layerarray = per_layer[ilayer]
        layerarray = layerarray.expand_dims({"layer": [ilayer]})
        layerarray.name= data_name
        result = xarray.combine_by_coords((result, layerarray), combine_attrs='no_conflicts')
    return result

def import_array(basedir, key, basename):
    idf_files = get_idf_filepaths(basedir+relative_paths[key], basename)
    layered_datset= convert_to_netcdf_per_layer(idf_files, basename)
    return layered_datset[basename]

basedir = "D:\\submodel\\SUBMODEL_working\\"

tops =import_array(basedir, "TOP", "TOP")
bots =import_array(basedir, "BOT", "BOT")


idomain = xarray.full_like(tops, dtype=np.int32, fill_value=1)

grid = mf6.VerticesDiscretization(tops, bots, idomain)

kd =import_array(basedir, "KD", "KD")
heigths = tops - bots
k =kd*(1.0/heigths)

i=0