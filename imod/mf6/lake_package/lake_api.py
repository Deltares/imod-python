from time import time_ns
import numpy as np
import xarray as xr

from imod import mf6

"""
This source file contains an interface to the lake package.
Usage: create instances of the LakeLake class, and optionally instances of the Outlets class,
and use the method "from_lakes_and_outlets" to create a lake package.
"""


missing_values = {
    "float32": np.nan,
    "int32": -6789012,
}

class LakeApi_Base:
    def __init__(self, number):
        self.object_number = number

    def get_times(self, timeseries_name):
        times = []
        if hasattr(self, timeseries_name):
            ts =getattr(self, timeseries_name)
            if type(ts)  == xr.DataArray:
                tstimes = [x for x in ts.coords["time"].values]
                times.extend(tstimes)
        return sorted(set(times))

    def has_transient_data(self, timeseries_name):
        if hasattr(self, timeseries_name):
            ts =getattr(self, timeseries_name)
            if type(ts)  == xr.DataArray:
                return True
        return False


    def add_timeseries_to_dataarray(self, timeseries_name, dataarray):
        current_object_data = dataarray.sel(index=self.object_number)
        if hasattr(self, timeseries_name):
            ts =getattr(self, timeseries_name)
            if ts is not None:
                index_of_object = dataarray.coords["index"].values.tolist().index(self.object_number)
                dataarray[{"index":index_of_object}]= current_object_data.combine_first(ts)
        return dataarray

class LakeLake(LakeApi_Base):
    """
    This class is used to initialize the lake package. It contains data relevant to 1 lake.
    The input needed to create an instance of this consists of a few scalars( name, starting stage)
    , some xarray data-arrays, and timeseries.
    The xarray data-arrays should be of the same size as the grid, and contain missing values in all
    locations where the lake does not exist ( similar to the input of the chd package)
    The missing value that you should use depends on the datatype and is contained in the missing_values
    dictionary.


    parameters:

        starting_stage: float,
        boundname: str,
        connectionType: xr.DataArray of integers.
        bed_leak: xr.DataArray of reals.
        top_elevation: xr.DataArray of reals.
        bot_elevation: xr.DataArray of reals.
        connection_length: xr.DataArray of reals.
        connection_width: xr.DataArray of reals.
        status,
        stage: timeseries of real numbers
        rainfall: timeseries of real numbers
        evaporation: timeseries of real numbers
        runoff: timeseries of real numbers
        inflow: timeseries of real numbers
        withdrawal: timeseries of real numbers
        auxiliary: timeseries of real numbers

    """
    timeseries_names = ["status", "stage", "rainfall", "evaporation", "runoff", "inflow", "withdrawal", "auxiliary"]

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
        status,
        stage,
        rainfall,
        evaporation,
        runoff,
        inflow,
        withdrawal,
        auxiliary,
    ):
        super().__init__( -1)
        self.starting_stage = starting_stage
        self.boundname = boundname
        self.connectionType = connectionType
        self.bed_leak = bed_leak
        self.top_elevation = top_elevation
        self.bottom_elevation = bot_elevation
        self.connection_length = connection_length
        self.connection_width = connection_width

        # timeseries
        self.status = status
        self.stage = stage
        self.rainfall = rainfall
        self.evaporation = evaporation
        self.runoff = runoff
        self.inflow = inflow
        self.withdrawal = withdrawal
        self.auxiliar = auxiliary

    @classmethod
    def get_subdomain_indices(cls, whole_domain_coords, subdomain_coords):
        """
        returns a vector of indices of the (real world) coordinates of subdomain_coords
        in whole_domain_coords.
        Example: if whole_domain_cords is (1.1, 2.2, 3.3, 4.4) and subdomain_cords is (2.2,3.3)
        then this function returns (1,2)
        """
        result = []
        list_whole_domain_coords = list(whole_domain_coords)
        for i in range(0, len(subdomain_coords)):
            result.append(list_whole_domain_coords.index(subdomain_coords[i]))
        return result

    @classmethod
    def get_1d_array(cls, grid_array):
        """
        this method takes as input an xarray defined over the whole domain. It has missing values
        on the locations where the lake does not exist. This function returns 4 1d arrays of equal length.
        They contain the row, column, layer and data values respectively, only at those locations where the values are not missing.
        """
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
        dummy = dummy.astype(grid_array.dtype)
        x_values = list(dummy.z.x.values)
        y_values = list(dummy.z.y.values)
        layer_values = list(dummy.z.layer.values)
        array_values = list(dummy.values)
        return x_values, y_values, layer_values, array_values


class OutletBase(LakeApi_Base):
    """
    Base class for the different kinds of outlets
    """
    timeseries_names = ["rate", "invert", "rough", "width", "slope"]

    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str):
        super().__init__( outletNumber)
        self.lake_in = lakeIn
        self.lake_out = lakeOut
        self.couttype = ""
        self.invert = -1
        self.width = -1
        self.roughness = -1
        self.slope = -1


class OutletManning(OutletBase):
    """
    This class represents a Manning Outlet
    """

    def __init__(
        self,
        outletNumber: int,
        lakeIn: str,
        lakeOut: str,
        invert: np.floating,
        width: np.floating,
        roughness: np.floating,
        slope: np.floating,
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
        invert: np.floating,
        width: np.floating,
    ):
        super().__init__(outletNumber, lakeIn, lakeOut)
        self.invert = invert
        self.width = width
        self.couttype = "WEIR"


class OutletSpecified(OutletBase):
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str, rate: np.floating):
        super().__init__(outletNumber, lakeIn, lakeOut)
        self.rate = rate
        self.couttype = "SPECIFIED"


def list_1d_to_xarray_1d(list, dimension_name):
    """
    This function creates a 1-dimensional xarray.DataArray with values
    equal to those in the input list. The coordinates are the indexes in the list.
    The dimension name is passed as a function argument.
    """
    nr_elem = len(list)
    dimensions = [dimension_name]
    coordinates = {dimension_name: range(0, nr_elem)}
    result = xr.DataArray(dims=dimensions, coords=coordinates)
    result.values = list
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
        prop = vars(outlet)[propertyname]
        if type(prop) == xr.DataArray:
            result_list.append(0.0)
        else:
            result_list.append(vars(outlet)[propertyname])
    return list_1d_to_xarray_1d(result_list, dimension_name)


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
    result_as_list = []
    for i in range(0, nrlakes):
        connection_prop = vars(list_of_lakes[i])[propertyname]
        _, _, _, prop_current_lake = LakeLake.get_1d_array(connection_prop)
        result_as_list += prop_current_lake
    return list_1d_to_xarray_1d(result_as_list, "connection_nr")


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
    return list_1d_to_xarray_1d(result_as_list, "lake_nr")


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
                result[i] = list_of_lakes[j].object_number
                break
        else:
            raise ValueError("could not find a lake with name {}".format(lakename))
    return result


def from_lakes_and_outlets(list_of_lakes, list_of_outlets=[],
        print_input=False,
        print_stage=False,
        print_flows=False,
        save_flows=False,
        stagefile=None,
        budgetfile=None,
        budgetcsvfile=None,
        package_convergence_filename=None,
        ts6_filename=None,
        time_conversion=None,
        length_conversion=None):
    """
    this function creates a lake_package given a list of lakes and optionally a list of outlets.
    """
    nrlakes = len(list_of_lakes)

    lakenumber = []
    row = []
    col = []
    layer = []

    for i in range(0, nrlakes):
        list_of_lakes[i].object_number = i + 1
        lyr, y, x, ctype = LakeLake.get_1d_array(list_of_lakes[i].connectionType)
        row += x
        col += y
        layer += lyr
        lakenumber += [list_of_lakes[i].object_number] * len(ctype)

    l_boundname = lake_list_lake_prop_to_xarray_1d(list_of_lakes, "boundname")
    l_starting_stage = lake_list_lake_prop_to_xarray_1d(list_of_lakes, "starting_stage")
    l_lakenr = list_1d_to_xarray_1d(list(range(1, nrlakes + 1)), "lake_nr")
    c_lakenumber = list_1d_to_xarray_1d(lakenumber, "connection_nr")
    c_row = list_1d_to_xarray_1d(row, "connection_nr")
    c_col = list_1d_to_xarray_1d(col, "connection_nr")
    c_layer = list_1d_to_xarray_1d(layer, "connection_nr")
    c_type = lake_list_connection_prop_to_xarray_1d(list_of_lakes, "connectionType")
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
            list_of_outlets, "couttype", "outlet_nr"
        )
        o_invert = outlet_list_prop_to_xarray_1d(list_of_outlets, "invert", "outlet_nr")
        o_roughness = outlet_list_prop_to_xarray_1d(
            list_of_outlets, "roughness", "outlet_nr"
        )
        o_width = outlet_list_prop_to_xarray_1d(list_of_outlets, "width", "outlet_nr")
        o_slope = outlet_list_prop_to_xarray_1d(list_of_outlets, "slope", "outlet_nr")
    ts_times =collect_all_times(list_of_lakes, list_of_outlets)
    ts_status = create_timeseries(list_of_lakes,ts_times, "status")
    ts_stage = create_timeseries(list_of_lakes,ts_times,"stage")
    ts_rainfall = create_timeseries(list_of_lakes,ts_times,"rainfall")
    ts_evaporation = create_timeseries(list_of_lakes,ts_times,"evaporation")
    ts_runoff = create_timeseries(list_of_lakes,ts_times,"runoff")
    ts_inflow = create_timeseries(list_of_lakes,ts_times,"inflow")
    ts_withdrawal  = create_timeseries(list_of_lakes,ts_times,"withdrwawal")
    ts_auxiliary  = create_timeseries(list_of_lakes,ts_times,"auxiliary")

    ts_rate = create_timeseries(list_of_outlets, ts_times, "rate")
    ts_invert = create_timeseries(list_of_outlets, ts_times, "invert")
    ts_rough = create_timeseries(list_of_outlets, ts_times, "rough")
    ts_width= create_timeseries(list_of_outlets, ts_times, "width")
    ts_slope= create_timeseries(list_of_outlets, ts_times, "slope")

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
        #timeseries (lake)
        ts_status,
        ts_stage,
        ts_rainfall,
        ts_evaporation ,
        ts_runoff,
        ts_inflow,
        ts_withdrawal,
        ts_auxiliary ,
        #timeseries (outlet)
        ts_rate ,
        ts_invert,
        ts_rough,
        ts_width,
        ts_slope,
        # options
        print_input,
        print_stage,
        print_flows,
        save_flows,
        stagefile,
        budgetfile,
        budgetcsvfile,
        package_convergence_filename,
        time_conversion,
        length_conversion)

    return result

def collect_all_times(  list_of_lakes, list_of_outlets):
    times=[]

    for timeseries_name in LakeLake.timeseries_names:
        for lake in list_of_lakes:
            if lake.has_transient_data(timeseries_name):
                times.extend(lake.get_times(timeseries_name))

    for timeseries_name in OutletBase.timeseries_names:
        for outlet in list_of_outlets:
            if outlet.has_transient_data(timeseries_name):
                times.extend(outlet.get_times(timeseries_name))

    return sorted(set(times))

def create_timeseries( list_of_lakes_or_outlets, ts_times,  timeseries_name):

    if not any(lake_or_outlet.has_transient_data(timeseries_name) for lake_or_outlet in list_of_lakes_or_outlets):
        return None

    object_numbers = []
    for lake_or_outlet in list_of_lakes_or_outlets:
        object_numbers.append(lake_or_outlet.object_number)

    dataarray = xr.DataArray( dims=("time", "index"), coords={"time": ts_times, "index": object_numbers},name = timeseries_name)
    for lake_or_outlet in list_of_lakes_or_outlets:
        dataarray = lake_or_outlet.add_timeseries_to_dataarray(timeseries_name, dataarray)
    return dataarray



