
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
 "KD":"KDW\\VERSION_1",
 "ANI":"ANI\\VERSION_1",
 "VCW": "VCW\\VERSION_1",
  }
output_dir = "d:\\tmp\\"
def get_idf_filepaths(dir, basename):
    '''
    returns a list of idf files present in a directory, specifically those that have a name starting with basename
    if basename is "KD" is might return KD1.IDF, KD2.idf etc.
    '''
    return  [Path(dir).joinpath(f) for f in listdir(dir) if isfile(join(dir, f)) and f.startswith(basename) and f.endswith(".IDF")]

def get_layer_number_from_filenpath(filepath):
    '''
    from a filepath like d:\\tmp\\KD13.IDF, returns 13.
    (This is used to assign IDF files to layers)
    '''
    filename = filepath.name
    numbers = re.findall(r'\d+', filename)
    if len(numbers)==1:
        return int(numbers[0])
    assert False, "could not find 1 number in filename"


def convert_to_netcdf_per_layer(idf_files, data_name):
    '''
    from a collection of idf files  (like KD1.idf, KD2.idf, KD3.idf)
    creates a netcdf dataset layers containing the data ( like the KD values in the IDF files)
    '''
    per_layer ={}
    for f in idf_files:
        layer = get_layer_number_from_filenpath(f)
        netcdf =idf.open(f)
        per_layer[layer]= netcdf

    nrlayers = len(per_layer)
    result = per_layer[1]
    result =result.expand_dims({"layer": [1]})
    result.name = data_name
    for ilayer in range(2, nrlayers+1):
        layerarray = per_layer[ilayer]
        layerarray = layerarray.expand_dims({"layer": [ilayer]})
        layerarray.name= data_name
        result = xarray.combine_by_coords((result, layerarray), combine_attrs='no_conflicts')
    return result

def import_array(basedir, key, basename):
    '''
    given a directory, the key in the relative path dictionary, and a root filename
    - opens the relative path wiht the given key relative to basedir
    -finds all idf files whose name starts with "basename"
    -associates each idf file with a layer
    -creates a dataset with a layer dimension, containing the data per layer that is in the idf files
    '''
    idf_files = get_idf_filepaths(basedir+relative_paths[key], basename)
    layered_datset= convert_to_netcdf_per_layer(idf_files, basename)
    return layered_datset[basename]

def create_mf6_grid(mf5_tops, mf5_bots):
    '''
    this function takes tops and bots as input layered according to the imod 5 convention:
    the layers are aquifers, with aquitards in between. So Layer 1 and Layer 2 can have some separation.
    This separatoin is an aquitard with vertical flow only.
    We use these arrays to create a modflow 6 grid. Here, the aquitards are layers as well, so
    this grid has much more layers than the imod 5 grid.

    modflow 5                    modflow 6
    layer number                 layer number
    -----------------------------------------------------
    layer 1                             1
    (aquitard is implicit)              2
     layer 2                            3
    (aquitard is implicit)              4

    '''

    mf6_tops = mf5_tops.sel(layer = 1)
    mf6_tops = mf6_tops.expand_dims({"layer": 1})
    mf6_tops = mf6_tops.rename("elevation")

    mf6_bots = mf5_bots.sel(layer = 1)
    mf6_bots = mf6_bots.expand_dims({"layer": 1})
    mf6_bots = mf6_bots.rename("elevation")

    mf5_nlayers = max(mf5_tops.coords["layer"]).values[()]
    mf6_nlayers = mf5_nlayers*2 -1
    top = 0
    bot = 0
    for ilayer in range(2, mf6_nlayers+1):
        if ilayer%2 ==0:
            top = mf5_bots.sel(layer = ilayer/2)
            bot = mf5_tops.sel(layer = ilayer/2+1)
        else:
            top = mf5_tops.sel(layer = (ilayer+1)/2)
            bot = mf5_bots.sel(layer = (ilayer+1)/2)
        top = top.rename("elevation")
        bot = bot.rename("elevation")
        top = top.assign_coords({"layer": (np.int64(ilayer))})
        top = top.expand_dims("layer")
        bot = bot.assign_coords({"layer": (np.int64(ilayer))})
        bot = bot.expand_dims("layer")


        mf6_tops = xarray.combine_by_coords((mf6_tops, top), combine_attrs='no_conflicts')
        mf6_bots = xarray.combine_by_coords((mf6_bots, bot), combine_attrs='no_conflicts')
    idomain = xarray.full_like(mf6_tops, fill_value=1, dtype=np.int32)
    idomain = idomain.rename({"elevation": "activity"})

    return mf6.VerticesDiscretization(mf6_tops["elevation"], mf6_bots["elevation"], idomain["activity"])

def create_K(grid):
    '''
    this function creates a hydraulic conductivity array with a K component ( horizontal flow)
    and a K33 component (vertical flow)
    The values of these arrays is based on modflow 5 idf files.
    We read these for KD (horizontal flow in aquifers) and VCW (vertical flow in aquitards)
    KD and VCW are converted to K.

    modflow 5                    modflow 6         horizontal K            vertical K
    layer number                 layer number      computed from:          computed from:
    -----------------------------------------------------
    layer 1                             1            KD (mf5 layer 1 )          KD (mf5 layer 1 )
    (aquitard is implicit)              2              0                        VCW(mf5 layer1)
     layer 2                            3            KD (mf5 layer 2 )          KD (mf5 layer 2 )
    (aquitard is implicit)              4                0                       VCW(mf5 layer2)

    '''


    mf6_heigths = grid.dataset["top"] - grid.dataset["bottom"]
    nrlayers = max(mf6_heigths.coords["layer"]).values[()]

    aquifer_layers = list(range(1,nrlayers+1,2))
    aquitard_layers =  list(range(2,nrlayers,2))

    mf5_heigths_aquifers = mf6_heigths.sel({"layer": aquifer_layers})
    mf5_heigths_aquitards = mf6_heigths.sel({"layer": aquitard_layers})
    mf5_heigths_aquifers=mf5_heigths_aquifers.assign_coords({"layer": (mf5_heigths_aquifers.coords["layer"]+1)/2})
    mf5_heigths_aquitards=mf5_heigths_aquitards.assign_coords({"layer":mf5_heigths_aquitards.coords["layer"]/2})

    kd =import_array(basedir, "KD", "KD")
    c = import_array(basedir, "VCW", "C")

    k_xx_aquifer = kd*(1.0/mf5_heigths_aquifers)
    k_xx_aquitard = xarray.zeros_like(c)

    k_zz_aquifer = k_xx_aquifer.copy(deep=True)
    k_zz_aquitard = mf5_heigths_aquitards*(1.0/c)

    mf5_k_layers = k_xx_aquifer.coords["layer"].values
    mf6_k_layers = mf5_k_layers*2 -1


    k_xx_aquifer = k_xx_aquifer.assign_coords({"layer": mf6_k_layers})
    k_zz_aquifer = k_zz_aquifer.assign_coords({"layer": mf6_k_layers})

    mf5_c_layers = c.coords["layer"].values
    mf6_c_layers = mf5_c_layers*2

    k_xx_aquitard = k_xx_aquitard.assign_coords({"layer": mf6_c_layers})
    k_zz_aquitard = k_zz_aquitard.assign_coords({"layer": mf6_c_layers})

    k_xx_aquifer = k_xx_aquifer.rename("K")
    k_xx_aquitard = k_xx_aquitard.rename("K")
    k_xx_new = merge_among_layer_dim(k_xx_aquifer, k_xx_aquitard)

    k_zz_aquifer = k_zz_aquifer.rename("K33")
    k_zz_aquitard = k_zz_aquitard.rename("K33")
    k_zz_new = merge_among_layer_dim(k_zz_aquifer, k_zz_aquitard)

    k_new = k_xx_new.merge(k_zz_new)
    return k_new

def merge_among_layer_dim(array1, array2):
    '''
    This function merges 2 dataArrays. They are supposed to have the same x and y coordinates, and
    different layer indices.
    This function is needed because the combine_by_coords function fails when the
    coordinates are "intermixed", for example when array1 holds the layers [1,3,5]
    and array2 holds the layers [2,4]
    '''
    layers_array_1 = array1.coords["layer"].values
    layers_array_2 = array2.coords["layer"].values
    nrlayers = max(max(layers_array_1), max(layers_array_2))
    for ilayer in range(1, nrlayers+1):
        layer = 0
        if ilayer in layers_array_1:
            layer = array1.sel(layer = ilayer )
        elif ilayer in layers_array_2:
            layer = array2.sel(layer = ilayer)
        else:
            assert False

        layer = layer.assign_coords({"layer": (np.int64(ilayer))})
        layer = layer.expand_dims("layer")

        if ilayer == 1:
            result = layer
        else:
            result = xarray.combine_by_coords((result, layer), combine_attrs='no_conflicts')
    return result

basedir = "D:\\submodel\\SUBMODEL_working\\"

tops =import_array(basedir, "TOP", "TOP")
bots =import_array(basedir, "BOT", "BOT")

grid = create_mf6_grid(tops, bots)
k_new = create_K(grid)
k_new.to_netcdf(output_dir + "knew.nc")
heigths = grid.dataset["top"] - grid.dataset["bottom"]
heigths.to_netcdf(output_dir + "heigths.nc")

i=0