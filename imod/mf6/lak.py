"""
This source file contains the Lake Package and interface objects to the lake
package. Usage: create instances of the LakeData class, and optionally
instances of the Outlets class, and use the method "from_lakes_and_outlets" to
create a lake package.
"""

import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from imod import mf6
from imod.mf6.pkgbase import BoundaryCondition, Package, PackageBase, VariableMetaData

class LakeApi_Base(PackageBase):
    """
    Base class for lake and outlet object.
    """

    def __init__(self, number):
        super().__init__()
        self.object_number = number

    def get_times(self, timeseries_name):
        """
        returns the times associated with the timeseries that has name "timeseries_name"
        """
        times = []
        vars = [k for k in self.dataset.keys()]  
        if timeseries_name in vars:              
            ts = self.dataset[timeseries_name]
            tstimes = [x for x in ts.coords["time"].values]
            times.extend(tstimes)
        return sorted(set(times))

    def has_transient_data(self, timeseries_name):
        """
        returns true if this object has transient data for a property with a given name
        (outlets can have either a scalar or a timeseries for some properties)
        """
        vars = [k for k in self.dataset.keys()]
        if timeseries_name in vars:
            ts = self.dataset[timeseries_name]
            if type(ts) == xr.DataArray:
                if "time"in ts.coords:
                    return True
        return False

    def add_timeseries_to_dataarray(self, timeseries_name, dataarray):
        """
        adds the timeseries of this lake or outlet to a data-array containing this timeseries for
        all lakes or outlets
        """
        current_object_data = dataarray.sel(index=self.object_number)
        if  timeseries_name in self.dataset:
            ts =self.dataset[timeseries_name]
            if ts is not None:
                index_of_object = (
                    dataarray.coords["index"].values.tolist().index(self.object_number)
                )
                dataarray[
                    {"index": index_of_object}
                ] = current_object_data.combine_first(ts)
        return dataarray
class LakeData(LakeApi_Base):
    """
    This class is used to initialize the lake package. It contains data
    relevant to one lake. The input needed to create an instance of this
    consists of a few scalars (name, starting stage), some xarray data-arrays,
    and time series. The xarray data-arrays should be of the same size as the
    grid, and contain missing values in all locations where the lake does not
    exist (similar to the input of the other grid based packages, such a
    ConstantHead, River, GeneralHeadBounandary, etc.).

    Parameters
    ----------

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
    timeseries_names = [
        "status",
        "stage",
        "rainfall",
        "evaporation",
        "runoff",
        "inflow",
        "withdrawal",
        "auxiliary",
    ]
    def __init__(
        self,
        starting_stage: float,
        boundname: str,
        connection_type,
        bed_leak=None,
        top_elevation=None,
        bot_elevation=None,
        connection_length=None,
        connection_width=None,
        status=None,
        stage=None,
        rainfall=None,
        evaporation=None,
        runoff=None,
        inflow=None,
        withdrawal=None,
        auxiliary=None,
    ):
        super().__init__(-1)
        self.dataset["starting_stage"] = starting_stage
        self.dataset["boundname"] = boundname
        self.dataset["connection_type"] = connection_type
        self.dataset["bed_leak"] = bed_leak
        self.dataset["top_elevation"] = top_elevation
        self.dataset["bottom_elevation"] = bot_elevation
        self.dataset["connection_length"] = connection_length
        self.dataset["connection_width"] = connection_width

        # timeseries data
        self.dataset["status"] = status
        self.dataset["stage"] = stage
        self.dataset["rainfall"] = rainfall
        self.dataset["evaporation"] = evaporation
        self.dataset["runoff"] = runoff
        self.dataset["inflow"] = inflow
        self.dataset["withdrawal"] = withdrawal
        self.dataset["auxiliary"] = auxiliary


class OutletBase(LakeApi_Base):
    """
    Base class for the different kinds of outlets
    """

    timeseries_names = ["rate", "invert", "rough", "width", "slope"]

    def __init__(self, outlet_number: int, lakein: str, lakeout: str):
        super().__init__(-1)        
        self.dataset = xr.Dataset()
        self.dataset["outlet_number"] = outlet_number
        self.object_number = outlet_number
        self.dataset["lakein"] = lakein
        self.dataset["lakeout"] = lakeout
        self.dataset["invert"] = None
        self.dataset["width"] = None
        self.dataset["roughness"] = None
        self.dataset["slope"] = None


class OutletManning(OutletBase):
    """
    Lake outlet which discharges via a rectangular outlet that uses Manning's
    equation.
    """

    _couttype = "manning"

    def __init__(
        self,
        outlet_number: int,
        lakein: str,
        lakeout: str,
        invert: np.floating,
        width: np.floating,
        roughness: np.floating,
        slope: np.floating,
    ):
        super().__init__(outlet_number, lakein, lakeout)
        self.dataset["invert"] = invert
        self.dataset["width"] = width
        self.dataset["roughness"] = roughness
        self.dataset["slope"] = slope


class OutletWeir(OutletBase):
    """
    Lake outlet which discharges via a sharp-crested weir.
    """

    _couttype = "weir"

    def __init__(
        self,
        outlet_number: int,
        lakein: str,
        lakeout: str,
        invert: np.floating,
        width: np.floating,
    ):
        super().__init__(outlet_number, lakein, lakeout)
        self.dataset["invert"] = invert
        self.dataset["width"] = width


class OutletSpecified(OutletBase):
    """
    Lake outlet which discharges a specified outflow.
    """

    _couttype = "specified"

    def __init__(
        self, outlet_number: int, lakein: str, lakeout: str, rate: np.floating
    ):
        super().__init__(outlet_number, lakein, lakeout)
        self.dataset["rate"] = rate

def collect_all_times(list_of_lakes, list_of_outlets):
    """
    all timeseries we pass to the lake package must have the same time discretization-otherwise the coordinates
    of the lake-package do not match. So in this function we collect all the time coordinates of all the lakes and
    outlets.
    """
    times = []

    for timeseries_name in LakeData.timeseries_names:
        for lake in list_of_lakes:
            if lake.has_transient_data(timeseries_name):
                times.extend(lake.get_times(timeseries_name))

    for timeseries_name in OutletBase.timeseries_names:
        for outlet in list_of_outlets:
            if outlet.has_transient_data(timeseries_name):
                times.extend(outlet.get_times(timeseries_name))
    return sorted(set(times))

def create_connection_data(lakes):
    connection_vars = [
        "bed_leak",
        "top_elevation",
        "bottom_elevation",
        "connection_length",
        "connection_width",
    ]

    # This will create a statically sized array of strings (dtype="<U10")
    claktype_string = np.array(
        [
            "horizontal",  # 0
            "vertical",  # 1
            "embeddedh",  # 2
            "embeddedv",  # 3
        ]
    )

    connection_data = defaultdict(list)
    cell_ids = []
    for lake in lakes:
        notnull = lake["connection_type"].notnull()
        indices = np.argwhere(notnull.values)
        xr_indices = {
            dim: xr.DataArray(index, dims=("cell_id",))
            for dim, index in zip(notnull.dims, indices.T)
        }

        # There should be no nodata values in connection_type, so we can use it to index.
        type_numeric = (
            lake.dataset["connection_type"].isel(**xr_indices).astype(int).values
        )
        type_string = claktype_string[type_numeric]
        connection_data["connection_type"].append(type_string)

        selection = lake.dataset[connection_vars].isel(**xr_indices)
        for var, da in selection.items():
            if not var.startswith("connection_"):
                var = f"connection_{var}"
            connection_data[var].append(da.values)

        cell_id = xr.DataArray(
            data=indices,
            coords={"celldim": list(xr_indices.keys())},
            dims=("boundary", "celldim"),
        )
        cell_ids.append(cell_id)

    connection_data = {
        k: xr.DataArray(data=np.concatenate(v), dims=["boundary"])
        for k, v in connection_data.items()
    }
    # Offset by one since MODFLOW is 1-based!
    connection_data["connection_cell_id"] = xr.concat(cell_ids, dim="boundary") + 1
    return connection_data


def create_outlet_data(outlets, name_to_number):
    outlet_vars = [
        "invert",
        "roughness",
        "width",
        "slope",
    ]
    outlet_data = defaultdict(list)
    for outlet in outlets:
        outlet_data["outlet_couttype"].append(outlet._couttype)

        # Convert names to numbers
        for var in ("lakein", "lakeout"):
            name = outlet.dataset[var].item()
            try:
                lake_number = name_to_number[name]
            except KeyError:
                names = ", ".join(name_to_number.keys())
                raise KeyError(
                    f"Outlet lake name {name} not found among lake names: {names}"
                )
            outlet_data[f"outlet_{var}"].append(lake_number)

        # For other values: fill in NaN if not applicable.
        for var in outlet_vars:
            if var in outlet.dataset   :
                if "time" in outlet.dataset[var].dims:
                    value = 0.0
                else:
                    value = outlet.dataset[var].item()
            else:
                value = np.nan
            outlet_data[f"outlet_{var}"].append(value)

    outlet_data = {
        k: xr.DataArray(data=v, dims=["outlet"]) for k, v in outlet_data.items()
    }
    return outlet_data


def create_timeseries(list_of_lakes_or_outlets, ts_times, timeseries_name):
    """
    In this function we create a dataarray with a given time coorridnate axis. We add all
    the timeseries of lakes or outlets with the given name. We also create a dimension to
    specify the lake or outlet number.
    """
    if not any(
        lake_or_outlet.has_transient_data(timeseries_name)
        for lake_or_outlet in list_of_lakes_or_outlets
    ):
        return None

    object_numbers = []
    for lake_or_outlet in list_of_lakes_or_outlets:
        object_numbers.append(lake_or_outlet.object_number)

    dataarray = xr.DataArray(
        dims=("time", "index"),
        coords={"time": ts_times, "index": object_numbers},
        name=timeseries_name,
    )
    for lake_or_outlet in list_of_lakes_or_outlets:
        dataarray = lake_or_outlet.add_timeseries_to_dataarray(
            timeseries_name, dataarray
        )
    return dataarray

class Period_internal:
    """
    The Period_internal class is used for rendering the lake package in jinja.
    There is no point in instantiating this class as a user.
    """

    def __init__(self, period_number):
        self.period_number = period_number
        self.nr_values = 0
        self.lake_or_outlet_number = []
        self.series_name = []
        self.value = []


class Lake(BoundaryCondition):
    """
    Lake (LAK) Package.

    Parameters
    ----------
    lake_number: array of integers (xr.DataArray)- dimension number of lakes:
        integer used as identifier for the lake.
    lake_starting_stage: array of floats (xr.DataArray)- dimension number of lakes:
        starting lake stage.
    lake_boundname:  array of strings (xr.DataArray)- dimension number of lakes:
        name of the lake

    connection_lake_number: array of floats (xr.DataArray)- dimension number of connections
        lake number for the current lake-to-aquifer connection.
    connection_cell_id: array of integers (xr.DataArray)- dimension number of connections

    connection_type: array of strings (xr.DataArray)- dimension number of connections
        indicates if connection is horizontal, vertical, embeddedH or embeddedV
    connection_bed_leak: array of floats (xr.DataArray)- dimension number of connections
        defines the bed leakance for the lake-GWF connection.
        BEDLEAK must be greater than or equal to zero or specified to be np.nan. If BEDLEAK is specified to
        be np.nan, the lake-GWF connection conductance is solely a function of aquifer properties in the
        connected GWF cell and lakebed sediments are assumed to be absent.
    connection_bottom_elevation: array of floats (xr.DataArray, optional)- dimension number of connections
        defines the bottom elevation for a horizontal lake-GWF connection.
        If not provided, will be set to the bottom elevation of the cell it is connected to.
    connection_top_elevation:array of floats (xr.DataArray, optional)- dimension number of connections
        defines the top elevation for a horizontal lake-GWF connection.
        If not provided, will be set to the top elevation of the cell it is connected to.
    connection_width: array of floats (xr.DataArray, optional)
        defines the connection face width for a horizontal lake-GWF connection.
        connwidth must be greater than zero for a horizontal lake-GWF connection. Any value can be
        specified if claktype is vertical, embeddedh, or embeddedv. If not set, will be set to dx or dy.
    connection_length: array of floats (xr.DataArray, optional)
        defines the distance between the connected GWF cellid node and the lake
        for a horizontal, embeddedh, or embeddedv lake-GWF connection. connlen must be greater than
        zero for a horizontal, embeddedh, or embeddedv lake-GWF connection. Any value can be specified
        if claktype is vertical. If not set, will be set to dx or dy.


    outlet_lakein: array of integers (xr.DataArray, optional)
        integer defining the lake number that outlet is connected to. Must be
        greater than zero.
    outlet_lakeout: array of integers (xr.DataArray, optional)
         integer value that defines the lake number that outlet discharge from lake outlet OUTLETNO
        is routed to. Must be greater than or equal to zero.
        If zero, outlet discharge from lake outlet OUTLETNO is discharged to an external
        boundary.
    outlet_couttype: array of string (xr.DataArray, optional)
        character string that defines the outlet type for the outlet OUTLETNO. Possible
        strings include: SPECIFIED–character keyword to indicate the outlet is defined as a specified
        flow. MANNING–character keyword to indicate the outlet is defined using Manning’s equation.
        WEIR–character keyword to indicate the outlet is defined using a sharp weir equation.
    outlet_invert: array of floats (xr.DataArray, optional):
        float or character value that defines the invert elevation for the lake outlet. A specified
        INVERT value is only used for active lakes if outlet_type for lake outlet OUTLETNO is not
        SPECIFIED.
    outlet_roughness: array of floats (xr.DataArray, optional)
        defines the roughness coefficient for the lake outlet. Any value can be specified
        if outlet_type is not MANNING.
    outlet_width: array of floats (xr.DataArray, optional)
        float or character value that defines the width of the lake outlet. A specified WIDTH value is
        only used for active lakes if outlet_type for lake outlet OUTLETNO is not SPECIFIED.
    outlet_slope: array of floats (xr.DataArray, optional)
        float or character value that defines the bed slope for the lake outlet. A specified SLOPE value is
        only used for active lakes if outlet_type for lake outlet OUTLETNO is MANNING.


        #time series (lake)
    ts_status: array of strings (xr.DataArray, optional)
        timeserie used to indicate lake status. Can be ACTIVE, INACTIVE, or CONSTANT.
        By default, STATUS is ACTIVE.
    ts_stage: array of floats (xr.DataArray, optional)
        timeserie used to specify the stage of the lake. The specified STAGE is only applied if
        the lake is a constant stage lake
    ts_rainfall: array of floats (xr.DataArray, optional)
        timeserie used to specify the rainfall rate (LT-1) for the lake. Value must be greater than or equal to zero.
    ts_evaporation: array of floats (xr.DataArray, optional)
        timeserie used to specify the  the maximum evaporation rate (LT-1) for the lake. Value must be greater than or equal to zero. I
    ts_runoff: array of floats (xr.DataArray, optional)
        timeserie used to specify the  the runoff rate (L3 T-1) for the lake. Value must be greater than or equal to zero.
    ts_inflow: array of floats (xr.DataArray, optional)
        timeserie used to specify the volumetric inflow rate (L3 T-1) for the lake. Value must be greater than or equal to zero.
    ts_withdrawal: array of floats (xr.DataArray, optional)
        timeserie used to specify the maximum withdrawal rate (L3 T-1) for the lake. Value must be greater than or equal to zero.
    ts_auxiliary: array of floats (xr.DataArray, optional)
        timeserie used to specify value of auxiliary variables of the lake

    #time series (outlet)
    ts_rate: array of floats (xr.DataArray, optional)
        timeserie used to specify the extraction rate for the lake outflow. A positive value indicates
        inflow and a negative value indicates outflow from the lake. RATE only applies to active
         (IBOUND > 0) lakes. A specified RATE is only applied if COUTTYPE for the OUTLETNO is SPECIFIED
    ts_invert: array of floats (xr.DataArray, optional)
        timeserie used to specify  the invert elevation for the lake outlet. A specified INVERT value is only used for
        active lakes if COUTTYPE for lake outlet OUTLETNO is not SPECIFIED.
    ts_rough: array of floats (xr.DataArray, optional)
        timeserie used to specify defines the roughness coefficient for the lake outlet. Any value can be
        specified if COUTTYPE is not MANNING.
    ts_width: array of floats (xr.DataArray, optional)
        timeserie used to specify  the width of the lake outlet. A specified WIDTH value is only used for active lakes if
        COUTTYPE for lake outlet OUTLETNO is not SPECIFIED.
    ts_slope: array of floats (xr.DataArray, optional)
        timeserie used to specify the bed slope for the lake outlet. A specified SLOPE value is only used for active lakes
        if COUTTYPE for lake outlet OUTLETNO is MANNING.


    print_input: ({True, False}, optional)
        keyword to indicate that the list of constant head information will
        be written to the listing file immediately after it is read. Default is
        False.
    print_stage: ({True, False}, optional)
        Keyword to indicate that the list of lake stages will be printed to the listing file for every
        stress period in which "HEAD PRINT" is specified in Output Control. If there is no Output Control
        option and PRINT_STAGE is specified, then stages are printed for the last time step of each
        stress period.
    print_flows: ({True, False}, optional)
        Indicates that the list of constant head flow rates will be printed to
        the listing file for every stress period time step in which "BUDGET
        PRINT" is specified in Output Control. If there is no Output Control
        option and PRINT FLOWS is specified, then flow rates are printed for the
        last time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that constant head flow terms will be written to the file
        specified with "BUDGET FILEOUT" in Output Control. Default is False.


    stagefile: (String, optional)
        name of the binary output file to write stage information.
    budgetfile: (String, optional)
        name of the binary output file to write budget information.
    budgetcsvfile: (String, optional)
        name of the comma-separated value (CSV) output file to write budget summary information.
        A budget summary record will be written to this file for each time step of the simulation.
    package_convergence_filename: (String, optional)
        name of the comma spaced values output file to write package convergence information.
    ts6_filename: String, optional
        defines a time-series file defining time series that can be used to assign time-varying values.
        See the "Time-Variable Input" section for instructions on using the time-series capability.
    time_conversion: float
        value that is used in converting outlet flow terms that use Manning’s equation or gravitational
        acceleration to consistent time units. TIME_CONVERSION should be set to 1.0, 60.0, 3,600.0,
        86,400.0, and 31,557,600.0 when using time units (TIME_UNITS) of seconds, minutes, hours, days,
        or years in the simulation, respectively. CONVTIME does not need to be specified if no lake
        outlets are specified or TIME_UNITS are seconds.,
    length_conversion: float
        float value that is used in converting outlet flow terms that use Manning’s equation or gravitational
        acceleration to consistent length units. LENGTH_CONVERSION should be set to 3.28081, 1.0, and 100.0
        when using length units (LENGTH_UNITS) of feet, meters, or centimeters in the simulation,
        respectively. LENGTH_CONVERSION does not need to be specified if no lake outlets are specified or
        LENGTH_UNITS are meters.
    """

    _pkg_id = "lak"
    _template = Package._initialize_template(_pkg_id)
    _metadata_dict = {
        "lake_number": VariableMetaData(np.integer),
        "lake_starting_stage": VariableMetaData(np.floating),
        "lake_boundname": VariableMetaData(np.str0),
        "connection_lake_number": VariableMetaData(np.integer),
        "connection_cell_id": VariableMetaData(np.integer),
        "connection_type": VariableMetaData(np.str0),
        "connection_bed_leak": VariableMetaData(np.floating),
        "connection_bottom_elevation": VariableMetaData(np.floating),
        "connection_top_elevation": VariableMetaData(np.floating),
        "connection_width": VariableMetaData(np.floating),
        "connection_length": VariableMetaData(np.floating),
        "outlet_lakein": VariableMetaData(np.integer),
        "outlet_lakeout": VariableMetaData(np.integer),
        "outlet_couttype": VariableMetaData(np.str0),
        "outlet_invert": VariableMetaData(np.floating),
        "outlet_roughness": VariableMetaData(np.floating),
        "outlet_width": VariableMetaData(np.floating),
        "outlet_slope": VariableMetaData(np.floating),
        "ts_status": VariableMetaData(np.str0),
        "ts_stage": VariableMetaData(np.floating),
        "ts_rainfall": VariableMetaData(np.floating),
        "ts_evaporation": VariableMetaData(np.floating),
        "ts_runoff": VariableMetaData(np.floating),
        "ts_inflow": VariableMetaData(np.floating),
        "ts_withdrawal": VariableMetaData(np.floating),
        "ts_rate": VariableMetaData(np.floating),
        "ts_invert": VariableMetaData(np.floating),
        "ts_rough": VariableMetaData(np.floating),
        "ts_width": VariableMetaData(np.floating),
        "ts_slope": VariableMetaData(np.floating),
    }
    _period_data = (
        "ts_status",
        "ts_stage",
        "ts_rainfall",
        "ts_evaporation",
        "ts_runoff",
        "ts_inflow",
        "ts_withdrawal",
        "ts_rate",
        "ts_invert",
        "ts_rough",
        "ts_width",
        "ts_slope",
    )

    def __init__(
        # lake
        self,
        lake_number,
        lake_starting_stage,
        lake_boundname,
        # connection
        connection_lake_number,
        connection_cell_id,
        connection_type,
        connection_bed_leak,
        connection_bottom_elevation,
        connection_top_elevation,
        connection_width,
        connection_length,
        # outlet
        outlet_lakein=None,
        outlet_lakeout=None,
        outlet_couttype=None,
        outlet_invert=None,
        outlet_roughness=None,
        outlet_width=None,
        outlet_slope=None,
        # time series (lake)
        ts_status=None,
        ts_stage=None,
        ts_rainfall=None,
        ts_evaporation=None,
        ts_runoff=None,
        ts_inflow=None,
        ts_withdrawal=None,
        ts_auxiliary=None,
        # time series (outlet)
        ts_rate=None,
        ts_invert=None,
        ts_rough=None,
        ts_width=None,
        ts_slope=None,
        # options
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
        length_conversion=None,
    ):
        super().__init__(locals())
        self.dataset["lake_boundname"] = lake_boundname
        self.dataset["lake_number"] = lake_number
        self.dataset["lake_starting_stage"] = lake_starting_stage

        self.dataset["connection_lake_number"] = connection_lake_number
        self.dataset["connection_cell_id"] = connection_cell_id
        self.dataset["connection_type"] = connection_type
        self.dataset["connection_bed_leak"] = connection_bed_leak
        self.dataset["connection_bottom_elevation"] = connection_bottom_elevation
        self.dataset["connection_top_elevation"] = connection_top_elevation
        self.dataset["connection_width"] = connection_width
        self.dataset["connection_length"] = connection_length

        self.dataset["outlet_lakein"] = outlet_lakein
        self.dataset["outlet_lakeout"] = outlet_lakeout
        self.dataset["outlet_couttype"] = outlet_couttype
        self.dataset["outlet_invert"] = outlet_invert
        self.dataset["outlet_roughness"] = outlet_roughness
        self.dataset["outlet_width"] = outlet_width
        self.dataset["outlet_slope"] = outlet_slope

        self.dataset["print_input"] = print_input
        self.dataset["print_stage"] = print_stage
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows

        self.dataset["stagefile"] = stagefile
        self.dataset["budgetfile"] = budgetfile
        self.dataset["budgetcsvfile"] = budgetcsvfile
        self.dataset["package_convergence_filename"] = package_convergence_filename
        self.dataset["ts6_filename"] = ts6_filename
        self.dataset["time_conversion"] = time_conversion
        self.dataset["length_conversion"] = length_conversion

        self.dataset["ts_status"] = ts_status
        self.dataset["ts_stage"] = ts_stage
        self.dataset["ts_rainfall"] = ts_rainfall
        self.dataset["ts_evaporation"] = ts_evaporation
        self.dataset["ts_runoff"] = ts_runoff
        self.dataset["ts_inflow"] = ts_inflow
        self.dataset["ts_withdrawal"] = ts_withdrawal
        self.dataset["ts_auxiliary"] = ts_auxiliary

        self.dataset["ts_rate"] = ts_rate
        self.dataset["ts_invert"] = ts_invert
        self.dataset["ts_rough"] = ts_rough
        self.dataset["ts_width"] = ts_width
        self.dataset["ts_slope"] = ts_slope

        self._pkgcheck()

    @staticmethod
    def from_lakes_and_outlets(
        lakes,
        outlets=None,
        print_input=False,
        print_stage=False,
        print_flows=False,
        save_flows=False,
        stagefile=None,
        budgetfile=None,
        budgetcsvfile=None,
        package_convergence_filename=None,
        time_conversion=None,
        length_conversion=None,
    ):
        package_content = {}
        name_to_number = {
            lake["boundname"].item(): i + 1 for i, lake in enumerate(lakes)
        }
        for lake in lakes:
            lake.object_number = name_to_number[lake["boundname"].values[()]]

        # Package data
        lake_numbers = list(name_to_number.values())
        n_connection = [lake["connection_type"].count() for lake in lakes]
        package_content["lake_starting_stage"] = xr.DataArray(
            data=[lake["starting_stage"].item() for lake in lakes],
            dims=["lake"],
        )
        package_content["lake_number"] = xr.DataArray(data=lake_numbers, dims=["lake"])
        package_content["lake_boundname"] = xr.DataArray(
            list(name_to_number.keys()), dims=["lake"]
        )

        # Connection data
        package_content["connection_lake_number"] = xr.DataArray(
            data=np.repeat(lake_numbers, n_connection),
            dims=["boundary"],
        )
        connection_data = create_connection_data(lakes)
        package_content.update(connection_data)

        ts_times = collect_all_times(lakes, outlets)
        package_content["ts_status"]  = create_timeseries(lakes, ts_times, "status")
        package_content["ts_stage"]  = create_timeseries(lakes, ts_times, "stage")
        package_content["ts_rainfall"]  = create_timeseries(lakes, ts_times, "rainfall")
        package_content["ts_evaporation"]  = create_timeseries(lakes, ts_times, "evaporation")
        package_content["ts_runoff"]  = create_timeseries(lakes, ts_times, "runoff")
        package_content["ts_inflow"]  = create_timeseries(lakes, ts_times, "inflow")
        package_content["ts_withdrawal"]  = create_timeseries(lakes, ts_times, "withdrwawal")
        package_content["ts_auxiliary"]  = create_timeseries(lakes, ts_times, "auxiliary")

        package_content["ts_rate"] = create_timeseries(outlets, ts_times, "rate")
        package_content["ts_invert"] = create_timeseries(outlets, ts_times, "invert")
        package_content["ts_rough"]= create_timeseries(outlets, ts_times, "rough")
        package_content["ts_width"] = create_timeseries(outlets, ts_times, "width")
        package_content["ts_slope"] = create_timeseries(outlets, ts_times, "slope")

        if outlets is not None:
            outlet_data = create_outlet_data(outlets, name_to_number)
            package_content.update(outlet_data)
        package_content["print_input"] = print_input
        package_content["print_stage"] = print_stage
        package_content["print_flows"] = print_flows
        package_content["save_flows"] = save_flows
        package_content["stagefile"] = stagefile
        package_content["budgetfile"] = budgetfile
        package_content["budgetcsvfile"] = budgetcsvfile
        package_content["package_convergence_filename"] = package_convergence_filename
        package_content["time_conversion"] = time_conversion
        package_content["length_conversion"] = length_conversion
        return mf6.Lake(**package_content)

    def _has_outlets(self):
        # item() will not work here if the object is an array.
        # .values[()] will simply return the full numpy array.
        outlet_lakein = self.dataset["outlet_lakein"].values[()]
        if outlet_lakein is None or (
            np.isscalar(outlet_lakein) and np.isnan(outlet_lakein)
        ):
            return False
        return True

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        for var in (
            "print_input",
            "print_stage",
            "print_flows",
            "save_flows",
            "stagefile",
            "budgetfile",
            "budgetcsvfile",
            "package_convergence_filename",
            "ts6_filename",
            "time_conversion",
            "length_conversion",
        ):
            value = self[var].item()
            if self._valid(value):
                d[var] = value

        d["nlakes"] = len(self.dataset["lake_number"])
        if self._has_outlets():
            d["noutlets"] = len(self.dataset["outlet_lakein"])
        else:
            d["noutlets"] = 0
        d["ntables"] = 0

        packagedata = []
        for name, number, stage in zip(
            self.dataset["lake_boundname"],
            self.dataset["lake_number"],
            self.dataset["lake_starting_stage"],
        ):
            nconn = (self.dataset["connection_lake_number"] == number).sum()
            row = tuple(a.item() for a in (number, stage, nconn, name))
            packagedata.append(row)
        d["packagedata"] = packagedata

        d["periods"] = []
        any_timeseries = any(
            [self._valid(self.dataset[name].values[()]) for name in self._period_data]
        )
        if any_timeseries:
            d["periods"] = self.create_period_data_block(globaltimes)
        d["nperiod"] = len(d["periods"])

        return self._template.render(d)

    def _connection_dataframe(self) -> pd.DataFrame:
        connection_vars = [
            "connection_lake_number",
            "connection_type",
            "connection_bed_leak",
            "connection_bottom_elevation",
            "connection_top_elevation",
            "connection_width",
            "connection_length",
        ]
        data_df = self.dataset[connection_vars].to_dataframe()
        data_df["iconn"] = np.arange(1, len(data_df) + 1)
        cell_id_df = self.dataset["connection_cell_id"].to_pandas()
        order = (
            ["connection_lake_number", "iconn"]
            + list(cell_id_df.columns)
            + connection_vars[1:]
        )
        return pd.concat([data_df, cell_id_df], axis=1)[order]

    def _outlet_dataframe(self) -> pd.DataFrame:
        outlet_vars = [
            "outlet_lakein",
            "outlet_lakeout",
            "outlet_couttype",
            "outlet_invert",
            "outlet_roughness",
            "outlet_width",
            "outlet_slope",
        ]
        df = self.dataset[outlet_vars].to_dataframe()
        return df

    def write_blockfile(self, directory, pkgname, globaltimes, binary):
        renderdir = pathlib.Path(directory.stem)
        content = self.render(
            directory=renderdir,
            pkgname=pkgname,
            globaltimes=globaltimes,
            binary=binary,
        )
        filename = directory / f"{pkgname}.{self._pkg_id}"
        with open(filename, "w") as f:
            f.write(content)
            f.write("\n")

            self._write_table_section(
                f,
                self._connection_dataframe(),
                "connectiondata",
            )

            if self._has_outlets():
                f.write("\n")
                self._write_table_section(f, self._outlet_dataframe(), "outlets")

        return

    def create_period_data_block(self, globaltimes):
        period_data_list = {}

        for timeseries_name in self._period_data:
            if self._valid(self.dataset[timeseries_name].values[()]):
                timeseries = self.dataset[timeseries_name]
                timeseries_times = [t for t in timeseries.coords["time"].values]
                timeseries_indexes = [i for i in timeseries.coords["index"].values]
                for itime in timeseries_times:
                    iperiod = np.where(globaltimes == itime)[0][0] + 1

                    for index in timeseries_indexes:
                        object_specific_data = timeseries.sel(
                            time=itime, index=index
                        ).values[()]
                        if not np.isnan(object_specific_data):
                            if iperiod not in period_data_list:
                                period_data_list[iperiod] = Period_internal(iperiod)
                            period_data_list[iperiod].nr_values += 1
                            period_data_list[iperiod].lake_or_outlet_number.append(
                                index
                            )
                            period_data_list[iperiod].series_name.append(
                                timeseries_name[3:]
                            )
                            period_data_list[iperiod].value.append(object_specific_data)
        keys = [x for x in period_data_list.keys()]
        keys = sorted(keys)
        return [period_data_list[k] for k in keys]

    def _package_data_to_sparse(self):
        return

    def fill_stress_perioddata(self):
        return

    def write_perioddata(self, directory, pkgname, binary):
        return
