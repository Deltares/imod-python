import itertools
from termios import VQUIT
from imod import mf6

import numpy as np
import xarray as xr

"""
This source file contains an interface to the lake package
"""

connection_types = {"HORIZONTAL": 0, "VERTICAL": 1, "EMBEDDEDH": 2, "EMBEDDEDV": 3}
missing_values = {
    "float32": np.nan,
    "int32": -6789012,
}


class LakeLake:
    """ """

    def __init__(
        self,
        starting_stage: float,
        boundname: str,
        connectionType,
        bed_leak,
        top_elevation,
        bot_elevation,
        connection_length,
        connection_width,
        laketable,
        status,
        stage,
        rainfall,
        evaporation,
        runoff,
        inflow,
        withdrawal,
        auxiliary,
    ):
        self.lake_number = -1
        self.starting_stage = starting_stage
        self.boundname = boundname
        self.connectionType = connectionType
        self.bed_leak = bed_leak
        self.top_elevation = top_elevation
        self.bottom_elevation = bot_elevation
        self.connection_length = connection_length
        self.connection_width = connection_width

        # table for this lake
        self.laketable = (laketable,)

        # timeseries
        self.status = (status,)
        self.stage = (stage,)
        self.rainfall = (rainfall,)
        self.evaporation = (evaporation,)
        self.runoff = (runoff,)
        self.inflow = (inflow,)
        self.withdrawal = (withdrawal,)
        self.auxiliar = auxiliary

    @classmethod
    def get_subdomain_indices(cls, whole_domain_coords, subdomain_coords):
        result = []
        list_whole_domain_coords = list(whole_domain_coords)
        for i in range(0, len(subdomain_coords)):
            result.append(list_whole_domain_coords.index(subdomain_coords[i]))
        return result

    @classmethod
    def get_1d_array(cls, grid_array):
        dummy = grid_array.sel()
        dummy = dummy.where(dummy != missing_values[grid_array.dtype.name], drop=True)
        x_indexes = cls.get_subdomain_indices(grid_array.x, dummy.x)
        y_indexes = cls.get_subdomain_indices(grid_array.y, dummy.y)
        layer_indexes = cls.get_subdomain_indices(grid_array.layer, dummy.layer)
        dummy = dummy.assign_coords(
            {"x": x_indexes, "y": y_indexes, "layer": layer_indexes}
        )

        dummy = dummy.stack(z=("x", "y", "layer"))
        dummy = dummy.dropna("z")

        x_values = list(dummy.z.x.values)
        y_values = list(dummy.z.y.values)
        layer_values = list(dummy.z.layer.values)
        array_values = list(dummy.values)
        return x_values, y_values, layer_values, array_values

        # todo: check input


class OutletBase:
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str):
        self.outletNumber = outletNumber
        self.lake_in = lakeIn
        self.lake_out = lakeOut
        self.couttype = ""
        self.invert = -1
        self.width = -1
        self.roughness = -1
        self.slope = -1

class OutletManning(OutletBase):
    def __init__(
        self,
        outletNumber: int,
        lakeIn: str,
        lakeOut: str,
        invert,
        width,
        roughness,
        slope,
    ):
        super().__init__(outletNumber, lakeIn, lakeOut)
        self.invert = invert
        self.width = width
        self.roughness = roughness
        self.slope = slope
        self.couttype = "MANNING"


class OutletWeir(OutletBase):
    def __init__(
        self,
        outletNumber: int,
        lakeIn: str,
        lakeOut: str,
        invert,
        width,
    ):
        super().__init__(outletNumber, lakeIn, lakeOut)
        self.invert = invert
        self.width = width
        self.couttype = "WEIR"

class OutletSpecified(OutletBase):
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str, rate):
        super().__init__(outletNumber, lakeIn, lakeOut)
        self.rate = rate
        self.couttype = "SPECIFIED"

class LakeTable:
    def __init__(self, stage, volume, surface, exchange_surface=None):
        self.stage = stage
        self.volume = volume
        self.surface = surface
        self.exchange_surface = exchange_surface
def list_1d_to_xarray_1d(list, dimension_name):
    nr_elem = len(list)
    dimensions= [dimension_name]
    coordinates = {dimension_name: range(0,nr_elem)}
    result = xr.DataArray(dims=dimensions, coords= coordinates)
    result.values = list
    return result


def outlet_list_prop_to_xarray_1d(list_of_outlets, propertyname, dimension_name):
    result_list =[]
    for outlet in list_of_outlets:
        result_list.append(vars(outlet)[propertyname])
    return list_1d_to_xarray_1d(result_list, dimension_name)

def lake_list_connection_prop_to_xarray_1d(list_of_lakes, propertyname):
    nrlakes = len(list_of_lakes)
    result_as_list = []
    for i in range(0, nrlakes):
        list_of_lakes[i].lake_number = i + 1
        connection_prop = vars(list_of_lakes[i])[propertyname]
        _, _, _, prop_current_lake = LakeLake.get_1d_array(connection_prop)
        result_as_list += prop_current_lake
    return list_1d_to_xarray_1d(result_as_list, "connection_nr")

def lake_list_lake_prop_to_xarray_1d(list_of_lakes, propertyname):
    nrlakes = len(list_of_lakes)
    result_as_list = [vars(list_of_lakes[i])[propertyname] for i in range(0, nrlakes)]
    return list_1d_to_xarray_1d(result_as_list, "lake_nr")

def map_names_to_lake_numbers(list_of_lakes, list_of_lakenames):
    result = [-1]*len(list_of_lakenames)
    for i in range (0, len(list_of_lakenames)):
        lakename = list_of_lakenames[i]
        for j in range(0, len(list_of_lakes)):
            if list_of_lakes[j].boundname == lakename:
                result[i] = list_of_lakes[j].lake_number
                break
        else:
            raise ValueError("could not find a lake with name {}".format(lakename))
    return result



def from_lakes_and_outlets(list_of_lakes, list_of_outlets=[]):

    nrlakes = len(list_of_lakes)

    lakenumber = []
    row = []
    col = []
    layer = []

    for i in range(0, nrlakes):

        layer, y, x, ctype = LakeLake.get_1d_array(
            list_of_lakes[i].connectionType
        )
        row += x
        col += y
        layer += layer
        lakenumber+= [list_of_lakes[i].lake_number]* len(ctype)
    l_boundname = lake_list_lake_prop_to_xarray_1d(list_of_lakes, "boundname")
    l_starting_stage = lake_list_lake_prop_to_xarray_1d(list_of_lakes, "starting_stage")
    l_lakenr = list_1d_to_xarray_1d(list(range(1, nrlakes+1)), "lake_nr")
    c_lakenumber = list_1d_to_xarray_1d(lakenumber, "connection_nr")
    c_row = list_1d_to_xarray_1d(row, "connection_nr")
    c_col = list_1d_to_xarray_1d(col, "connection_nr")
    c_layer = list_1d_to_xarray_1d(layer, "connection_nr")
    c_type = lake_list_connection_prop_to_xarray_1d(list_of_lakes, "connectionType")
    c_bed_leak = lake_list_connection_prop_to_xarray_1d(list_of_lakes, "bed_leak")
    c_bottom_elevation  = lake_list_connection_prop_to_xarray_1d(list_of_lakes, "bottom_elevation")
    c_top_elevation   = lake_list_connection_prop_to_xarray_1d(list_of_lakes, "top_elevation")
    c_width = lake_list_connection_prop_to_xarray_1d(list_of_lakes, "connection_width")
    c_length = lake_list_connection_prop_to_xarray_1d(list_of_lakes, "connection_length")
    o_lakein_str = outlet_list_prop_to_xarray_1d(list_of_outlets, "lake_in", "outlet_nr")
    o_lakeout_str =  outlet_list_prop_to_xarray_1d(list_of_outlets, "lake_out", "outlet_nr")
    o_lakein = map_names_to_lake_numbers(list_of_lakes, o_lakein_str)
    o_lakeout = map_names_to_lake_numbers(list_of_lakes, o_lakeout_str)
    o_couttype = outlet_list_prop_to_xarray_1d(list_of_outlets, "couttype", "outlet_nr")
    o_invert = outlet_list_prop_to_xarray_1d(list_of_outlets, "invert", "outlet_nr")
    o_roughness= outlet_list_prop_to_xarray_1d(list_of_outlets, "roughness", "outlet_nr")
    o_width = outlet_list_prop_to_xarray_1d(list_of_outlets, "width", "outlet_nr")
    o_slope = outlet_list_prop_to_xarray_1d(list_of_outlets, "slope", "outlet_nr")

    result = mf6.Lake(        # lake
        l_lakenr,
        l_starting_stage,
        l_boundname,
        # connection
        c_lakenumber,
        c_row,
        c_col,
        c_layer,
        c_type,
        c_bed_leak,
        c_bottom_elevation,
        c_top_elevation,
        c_width,
        c_length,
        # outlet
        o_lakein,
        o_lakeout,
        o_couttype,
        o_invert,
        o_roughness,
        o_width,
        o_slope)
    return result





