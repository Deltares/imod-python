import numpy as np
import xarray as xr

from imod import mf6

"""
This source file contains an interface to the lake package.
Usage: create instances of the LakeLake class, and optionally instances of the Outlets class,
and use the method "from_lakes_and_outlets" to create a lake package.
"""


class LakeLake:
    """
    This class is used to initialize the lake package. It contains data relevant to 1 lake.
    The input needed to create an instance of this consists of a few scalars( name, starting stage)
    , some xarray data-arrays, and timeseries.
    The xarray data-arrays should be of the same size as the grid, and contain missing values in all
    locations where the lake does not exist ( similar to the input of the chd package)

    parameters:

        starting_stage: float,
        boundname: str,
        connection_type: xr.DataArray of integers.
        bed_leak: xr.DataArray of floats.
        top_elevation: xr.DataArray of floats.
        bot_elevation: xr.DataArray of floats.
        connection_length: xr.DataArray of floats.
        connection_width: xr.DataArray of floats.
        status,
        stage: timeseries of float numbers
        rainfall: timeseries of float numbers
        evaporation: timeseries of float numbers
        runoff: timeseries of float numbers
        inflow: timeseries of float numbers
        withdrawal: timeseries of float numbers
        auxiliary: timeseries of float numbers

    """

    def __init__(
        self,
        starting_stage: float,
        boundname: str,
        connection_type,
        bed_leak,
        top_elevation,
        bot_elevation,
        connection_length,
        connection_width,
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
        self.connection_type = connection_type
        self.bed_leak = bed_leak
        self.top_elevation = top_elevation
        self.bottom_elevation = bot_elevation
        self.connection_length = connection_length
        self.connection_width = connection_width

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
    def get_1d_array(cls, grid_array):
        """
        this method takes as input an xarray defined over the whole domain. It has missing values
        on the locations where the lake does not exist. This function returns 4 1d arrays of equal length.
        They contain the row, column, layer and data values respectively, only at those locations where the values are not missing.
        """
        dummy = grid_array.sel()

        dummy.coords["x"] = np.arange(1, len(dummy.x) + 1)
        dummy.coords["y"] = np.arange(1, len(dummy.y) + 1)
        dummy = dummy.where(dummy.notnull(), drop=True)
        dummy = dummy.stack(z=("x", "y", "layer"))
        dummy = dummy.dropna("z")

        x_values = dummy.x.values
        y_values = dummy.y.values
        layer_values = dummy.layer.values
        array_values = dummy.values
        return x_values, y_values, layer_values, array_values


class OutletBase:
    """
    Base class for the different kinds of outlets
    """

    def __init__(self, outlet_number: int, lake_in: str, lake_out: str):
        self.outlet_number = outlet_number
        self.lake_in = lake_in
        self.lake_out = lake_out
        self.invert = -1
        self.width = -1
        self.roughness = -1
        self.slope = -1


class OutletManning(OutletBase):
    """
    This class represents a Manning Outlet
    """

    _couttype = "manning"

    def __init__(
        self,
        outlet_number: int,
        lake_in: str,
        lake_out: str,
        invert: np.floating,
        width: np.floating,
        roughness: np.floating,
        slope: np.floating,
    ):
        super().__init__(outlet_number, lake_in, lake_out)
        self.invert = invert
        self.width = width
        self.roughness = roughness
        self.slope = slope


class OutletWeir(OutletBase):
    _couttype = "weir"

    def __init__(
        self,
        outlet_number: int,
        lake_in: str,
        lake_out: str,
        invert: np.floating,
        width: np.floating,
    ):
        super().__init__(outlet_number, lake_in, lake_out)
        self.invert = invert
        self.width = width


class OutletSpecified(OutletBase):
    _couttype = "specified"

    def __init__(
        self, outlet_number: int, lake_in: str, lake_out: str, rate: np.floating
    ):
        super().__init__(outlet_number, lake_in, lake_out)
        self.rate = rate


def nparray_to_xarray_1d(numpy_array_1d, dimension_name):
    """
    This function creates a 1-dimensional xarray.DataArray with values
    equal to those in the numpy array. The coordinates are the indexes in the array.
    The dimension name is passed as a function argument.
    """
    nr_elem = len(numpy_array_1d)
    dimensions = [dimension_name]
    coordinates = {dimension_name: range(0, nr_elem)}
    result = xr.DataArray(dims=dimensions, coords=coordinates)
    result.values = numpy_array_1d
    return result


def outlet_list_prop_to_xarray_1d(list_of_outlets, propertyname, dimension_name):
    """
    given the list of outlets and the name of a property, it creates an xarray.DataArray with
    all the properties appended in one list.
    For example, if outlet_1 has a slope of 3.0 and outlet_2 has a slope of 4.0, it returns an 1d xarray.DataArray
    containing the values (3.0, 4.0)
    The sole dimension is given the name dimension_name, and the coordinates are the indexes.
    """
    result_list = []
    for outlet in list_of_outlets:
        if propertyname in outlet.__dict__.keys():
            result_list.append(vars(outlet)[propertyname])
        else:
            result_list.append(getattr(outlet, propertyname))
    return nparray_to_xarray_1d(result_list, dimension_name)


def lake_list_connection_prop_to_xarray_1d(list_of_lakes, propertyname):
    """
    given the list of lakes and the name of a property, it creates an xarray.DataArray with
    all the properties appended in one list.
    This function is specifically for those properties that are lists (in practice, the connection properties)
    For example, if lake_1 has 2 connections with tops of  (3.0, 4.0) and lake_2 has 2 connections with tops of  (6.0, 7.0) , it returns an 1d xarray.DataArray
    containing the values (3.0, 4.0, 6.0,7.0)
    The sole dimension is given the name dimension_name, and the coordinates are the indexes.
    """
    nrlakes = len(list_of_lakes)
    result = np.array([])
    for i in range(0, nrlakes):
        connection_prop = vars(list_of_lakes[i])[propertyname]
        _, _, _, prop_current_lake = LakeLake.get_1d_array(connection_prop)
        result = np.append(result, prop_current_lake)
    return nparray_to_xarray_1d(result, "connection_nr")


def lake_list_lake_prop_to_xarray_1d(list_of_lakes, propertyname):
    """
    given the list of lakes and the name of a property, it creates an xarray.DataArray with
    all the properties appended in one list.
    This function is specifically for those properties that are scalars (in practice, the properties of the lake itself)
    For example, if lake_1 has a starting_stage of (3.0) and lake_2 has a starting_stage  of  (6.0) , it returns an 1d xarray.DataArray
    containing the values (3.0,6.0)
    The sole dimension is given the name dimension_name, and the coordinates are the indexes.
    """
    nrlakes = len(list_of_lakes)
    result_as_list = [vars(list_of_lakes[i])[propertyname] for i in range(0, nrlakes)]
    return nparray_to_xarray_1d(result_as_list, "lake_nr")


def map_names_to_lake_numbers(list_of_lakes, list_of_lakenames):
    """
    given a list of lakes, and a list of lakenames, it generates a list of the indexes of the lakenames
    in the list of lakes.
    For example, if lake_1.name = "Naardermeer" and lake_2.name = "Ijsselmeer"
    and we get the list_of_lakenames = ("Naardermeer", "Naardermeer", "Ijsselmeer")
    then the return value is (0,0,1)
    """

    result = [-1] * len(list_of_lakenames)
    for i in range(0, len(list_of_lakenames)):
        lakename = list_of_lakenames[i]
        for j in range(0, len(list_of_lakes)):
            if list_of_lakes[j].boundname == lakename:
                result[i] = list_of_lakes[j].lake_number
                break
        else:
            raise ValueError("could not find a lake with name {}".format(lakename))
    return result


def from_lakes_and_outlets(list_of_lakes, list_of_outlets=[]):
    """
    this function creates a lake_package given a list of lakes and optionally a list of outlets.
    """
    nrlakes = len(list_of_lakes)

    lakenumber = np.array([])
    row = np.array([])
    col = np.array([])
    layer = np.array([])

    for i in range(0, nrlakes):
        list_of_lakes[i].lake_number = i + 1
        lyr, y, x, ctype = LakeLake.get_1d_array(list_of_lakes[i].connection_type)
        row = np.append(row, x)
        col = np.append(col, y)
        layer = np.append(layer, lyr)
        lakenumber = np.append(lakenumber, [list_of_lakes[i].lake_number] * len(ctype))

    l_boundname = lake_list_lake_prop_to_xarray_1d(list_of_lakes, "boundname")
    l_starting_stage = lake_list_lake_prop_to_xarray_1d(list_of_lakes, "starting_stage")
    l_lakenr = nparray_to_xarray_1d(list(range(1, nrlakes + 1)), "lake_nr")
    c_lakenumber = nparray_to_xarray_1d(lakenumber, "connection_nr")
    c_row = nparray_to_xarray_1d(row, "connection_nr")
    c_col = nparray_to_xarray_1d(col, "connection_nr")
    c_layer = nparray_to_xarray_1d(layer, "connection_nr")
    c_type = lake_list_connection_prop_to_xarray_1d(list_of_lakes, "connection_type")
    c_bed_leak = lake_list_connection_prop_to_xarray_1d(list_of_lakes, "bed_leak")
    c_bottom_elevation = lake_list_connection_prop_to_xarray_1d(
        list_of_lakes, "bottom_elevation"
    )
    c_top_elevation = lake_list_connection_prop_to_xarray_1d(
        list_of_lakes, "top_elevation"
    )
    c_width = lake_list_connection_prop_to_xarray_1d(list_of_lakes, "connection_width")
    c_length = lake_list_connection_prop_to_xarray_1d(
        list_of_lakes, "connection_length"
    )
    o_lakein_str = outlet_list_prop_to_xarray_1d(
        list_of_outlets, "lake_in", "outlet_nr"
    )
    o_lakeout_str = outlet_list_prop_to_xarray_1d(
        list_of_outlets, "lake_out", "outlet_nr"
    )
    o_lakein = None
    o_lakeout = None
    o_couttype = None
    o_invert = None
    o_roughness = None
    o_width = None
    o_slope = None
    if len(list_of_outlets) > 0:
        o_lakein = map_names_to_lake_numbers(list_of_lakes, o_lakein_str)
        o_lakeout = map_names_to_lake_numbers(list_of_lakes, o_lakeout_str)
        o_couttype = outlet_list_prop_to_xarray_1d(
            list_of_outlets, "_couttype", "outlet_nr"
        )
        o_invert = outlet_list_prop_to_xarray_1d(list_of_outlets, "invert", "outlet_nr")
        o_roughness = outlet_list_prop_to_xarray_1d(
            list_of_outlets, "roughness", "outlet_nr"
        )
        o_width = outlet_list_prop_to_xarray_1d(list_of_outlets, "width", "outlet_nr")
        o_slope = outlet_list_prop_to_xarray_1d(list_of_outlets, "slope", "outlet_nr")

    result = mf6.Lake(  # lake
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
        o_slope,
    )
    return result
