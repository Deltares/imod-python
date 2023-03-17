"""
This source file contains the Lake Package and interface objects to the lake
package. Usage: create instances of the LakeData class, and optionally
instances of the Outlets class, and use the method "from_lakes_and_outlets" to
create a lake package.
"""

import pathlib
import textwrap
from collections import defaultdict

import jinja2
import numpy as np
import pandas as pd
import xarray as xr

from imod import mf6
from imod.mf6.pkgbase import BoundaryCondition, Package, PackageBase
from imod.schemata import AllValueSchema, DimsSchema, DTypeSchema

CONNECTION_DIM = "connection_dim"
OUTLET_DIM = "outlet_dim"
LAKE_DIM = "lake_dim"


class LakeApi_Base(PackageBase):
    """
    Base class for lake and outlet object.
    """

    def __init__(self):
        super().__init__()


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
        lake_table=None
    ):
        super().__init__()
        self.dataset["starting_stage"] = starting_stage
        self.dataset["boundname"] = boundname
        self.dataset["connection_type"] = connection_type
        self.dataset["bed_leak"] = bed_leak
        self.dataset["top_elevation"] = top_elevation
        self.dataset["bottom_elevation"] = bot_elevation
        self.dataset["connection_length"] = connection_length
        self.dataset["connection_width"] = connection_width

        self.dataset["lake_table"] = None
        if lake_table is not None:
            self.dataset["lake_table"] = lake_table

        # timeseries data

        times = []
        timeseries_dict = {
            "status": status,
            "stage": stage,
            "rainfall": rainfall,
            "evaporation": evaporation,
            "runoff": runoff,
            "inflow": inflow,
            "withdrawal": withdrawal,
            "auxiliary": auxiliary,
        }
        for _, timeseries in timeseries_dict.items():
            if timeseries is not None:
                if "time" in timeseries.coords:
                    times.extend([x for x in timeseries.coords["time"].values])
        times = sorted(set(times))
        self.dataset.assign_coords({"time": times})
        for ts_name, ts_object in timeseries_dict.items():
            if ts_object is not None:
                fillvalue = np.nan
                if not pd.api.types.is_numeric_dtype(ts_object.dtype):
                    fillvalue = ""
                self.dataset[ts_name] = ts_object.reindex(
                    {"time": times}, fill_value=fillvalue
                )
            else:
                self.dataset[ts_name] = None


class OutletBase(LakeApi_Base):
    """
    Base class for the different kinds of outlets
    """

    timeseries_names = ["rate", "invert", "rough", "width", "slope"]

    def __init__(self, lakein: str, lakeout: str):
        super().__init__()
        self.dataset = xr.Dataset()
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
        lakein: str,
        lakeout: str,
        invert,
        width,
        roughness,
        slope,
    ):
        super().__init__(lakein, lakeout)
        self.dataset["invert"] = invert
        self.dataset["width"] = width
        self.dataset["roughness"] = roughness
        self.dataset["slope"] = slope


class OutletWeir(OutletBase):
    """
    Lake outlet which discharges via a sharp-crested weir.
    """

    _couttype = "weir"

    def __init__(self, lakein: str, lakeout: str, invert, width):
        super().__init__(lakein, lakeout)
        self.dataset["invert"] = invert
        self.dataset["width"] = width


class OutletSpecified(OutletBase):
    """
    Lake outlet which discharges a specified outflow.
    """

    _couttype = "specified"

    def __init__(self, lakein: str, lakeout: str, rate):
        super().__init__(lakein, lakeout)
        self.dataset["rate"] = rate


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
        indices = np.argwhere(
            notnull.values
        )  # -1 to convert mf6's 1 based indexing to np's 0-based indexing
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

        # Offset by one since MODFLOW is 1-based!
        cell_id = xr.DataArray(
            data=indices + 1,
            coords={"celldim": list(xr_indices.keys())},
            dims=(CONNECTION_DIM, "celldim"),
        )
        cell_ids.append(cell_id)

    connection_data = {
        k: xr.DataArray(data=np.concatenate(v), dims=[CONNECTION_DIM])
        for k, v in connection_data.items()
    }

    connection_data["connection_cell_id"] = xr.concat(cell_ids, dim=CONNECTION_DIM)
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
            if var in outlet.dataset:
                if "time" in outlet.dataset[var].dims:
                    value = 0.0
                else:
                    value = outlet.dataset[var].item()
                if value is None:
                    value = np.nan
            else:
                value = np.nan
            outlet_data[f"outlet_{var}"].append(value)

    outlet_data = {
        k: xr.DataArray(data=v, dims=[OUTLET_DIM]) for k, v in outlet_data.items()
    }
    return outlet_data


def concatenate_timeseries(list_of_lakes_or_outlets, timeseries_name):
    """
    In this function we create a dataarray with a given time coorridnate axis. We add all
    the timeseries of lakes or outlets with the given name. We also create a dimension to
    specify the lake or outlet number.
    """
    if list_of_lakes_or_outlets is None:
        return None

    list_of_dataarrays = []
    list_of_indices = []
    for index, lake_or_outlet in enumerate(list_of_lakes_or_outlets):
        if timeseries_name in lake_or_outlet.dataset:
            da = lake_or_outlet.dataset[timeseries_name]
            if "time" in da.coords:
                list_of_dataarrays.append(da)
                list_of_indices.append(index + 1)

        index = index + 1
    if len(list_of_dataarrays) == 0:
        return None
    fill_value = np.nan
    if not pd.api.types.is_numeric_dtype(list_of_dataarrays[0].dtype):
        fill_value = ""
    out = xr.concat(
        list_of_dataarrays, join="outer", dim="index", fill_value=fill_value
    )
    out = out.assign_coords(index=list_of_indices)
    return out

    
def join_lake_tables(lake_numbers,  lakes):
    nr_lakes = len(lakes)
    assert len(lake_numbers) == nr_lakes

    any_lake_table = any([lake["lake_table"] is not None for lake in lakes])
    if not any_lake_table:
        return None

    lake_tables = []
    for i in range(nr_lakes):
        if lakes[i]["lake_table"] is not None:
           lake_number = lake_numbers[i]
           lakes[i]["lake_table"] = lakes[i]["lake_table"].expand_dims(dim = {"lake_nr":[lake_number]})
           lake_tables.append(lakes[i]["lake_table"].copy(deep=True))
        
    result = xr.merge(lake_tables, compat='no_conflicts')
    return result["lake_table"]



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

    lake_tables:  array of floats (xr.DataArray, optional)
        the array contains the lake tables- for those lakes that have them


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
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "lak"
    _template = Package._initialize_template(_pkg_id)

    _period_data_lakes = (
        "ts_status",
        "ts_stage",
        "ts_rainfall",
        "ts_evaporation",
        "ts_runoff",
        "ts_inflow",
        "ts_withdrawal",
        "ts_auxiliary",
    )
    _period_data_outlets = (
        "ts_rate",
        "ts_invert",
        "ts_rough",
        "ts_width",
        "ts_slope",
    )

    _period_data = _period_data_lakes + _period_data_outlets

    _init_schemata = {
        "lake_number": [DTypeSchema(np.integer), DimsSchema(LAKE_DIM)],
        "lake_starting_stage": [DTypeSchema(np.floating), DimsSchema(LAKE_DIM)],
        "lake_boundname": [DTypeSchema(str), DimsSchema(LAKE_DIM)],
        "connection_lake_number": [DTypeSchema(np.integer), DimsSchema(CONNECTION_DIM)],
        "connection_type": [DTypeSchema(str), DimsSchema(CONNECTION_DIM)],
        "connection_bed_leak": [DTypeSchema(np.floating), DimsSchema(CONNECTION_DIM)],
        "connection_bottom_elevation": [
            DTypeSchema(np.floating),
            DimsSchema(CONNECTION_DIM),
        ],
        "connection_top_elevation": [
            DTypeSchema(np.floating),
            DimsSchema(CONNECTION_DIM),
        ],
        "connection_width": [DTypeSchema(np.floating), DimsSchema(CONNECTION_DIM)],
        "connection_length": [DTypeSchema(np.floating), DimsSchema(CONNECTION_DIM)],
        "outlet_lakein": [
            DTypeSchema(np.integer),
            DimsSchema(OUTLET_DIM) | DimsSchema(),
        ],
        "outlet_lakeout": [
            DTypeSchema(np.integer),
            DimsSchema(OUTLET_DIM) | DimsSchema(),
        ],
        "outlet_couttype": [DTypeSchema(str), DimsSchema(OUTLET_DIM) | DimsSchema()],
        "outlet_invert": [
            DTypeSchema(np.floating),
            DimsSchema(OUTLET_DIM) | DimsSchema(),
        ],
        "outlet_roughness": [
            DTypeSchema(np.floating),
            DimsSchema(OUTLET_DIM) | DimsSchema(),
        ],
        "outlet_width": [
            DTypeSchema(np.floating),
            DimsSchema(OUTLET_DIM) | DimsSchema(),
        ],
        "outlet_slope": [
            DTypeSchema(np.floating),
            DimsSchema(OUTLET_DIM) | DimsSchema(),
        ],
        "ts_status": [DTypeSchema(str), DimsSchema("index", "time") | DimsSchema()],
        "ts_stage": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_rainfall": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_evaporation": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_runoff": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_inflow": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_withdrawal": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_auxiliary": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_rate": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_invert": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_rough": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_width": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
        "ts_slope": [
            DTypeSchema(np.floating),
            DimsSchema("index", "time") | DimsSchema(),
        ],
    }

    _write_schemata = {
        "lake_number": [AllValueSchema(">", 0)],
        "connection_lake_number": [AllValueSchema(">", 0)],
        "connection_cell_id": [AllValueSchema(">", 0)],
        "connection_width": [AllValueSchema(">", 0)],
        "connection_length": [AllValueSchema(">", 0)],
        "outlet_lakein": [AllValueSchema(">", 0)],
        "outlet_lakeout": [AllValueSchema(">", 0)],
        "outlet_width": [AllValueSchema(">", 0)],
    }

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
        # lake tables
        lake_tables = None,
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
        validate=True,
    ):
        super().__init__(locals())
        self.dataset["lake_boundname"] = lake_boundname
        self.dataset["lake_number"] = lake_number
        self.dataset["lake_starting_stage"] = lake_starting_stage

        nr_indices = int(self.dataset["lake_number"].data.max())
        if outlet_lakein is not None:
            nroutlets = len(outlet_lakein.data)
            nr_indices = max(nr_indices, nroutlets)

        self.dataset = self.dataset.assign_coords(index=range(1, nr_indices + 1, 1))

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
        if ts_status is not None:
            self.dataset["ts_status"] = self._convert_to_string_dataarray(
                self.dataset["ts_status"]
            )
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

        self.dataset["lake_tables"] = lake_tables

        self._validate_init_schemata(validate)

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

        # Package data
        lake_numbers = list(name_to_number.values())
        n_connection = [lake["connection_type"].count().values[()] for lake in lakes]
        package_content["lake_starting_stage"] = xr.DataArray(
            data=[lake["starting_stage"].item() for lake in lakes],
            dims=[LAKE_DIM],
        )
        package_content["lake_number"] = xr.DataArray(
            data=lake_numbers, dims=[LAKE_DIM]
        )
        package_content["lake_boundname"] = xr.DataArray(
            list(name_to_number.keys()), dims=[LAKE_DIM]
        )

        # Connection data
        package_content["connection_lake_number"] = xr.DataArray(
            data=np.repeat(lake_numbers, n_connection),
            dims=[CONNECTION_DIM],
        )
        connection_data = create_connection_data(lakes)
        package_content.update(connection_data)
        for ts_name in Lake._period_data_lakes:
            shortname = ts_name[3:]
            package_content[ts_name] = concatenate_timeseries(lakes, shortname)

        for ts_name in Lake._period_data_outlets:
            shortname = ts_name[3:]
            package_content[ts_name] = concatenate_timeseries(outlets, shortname)

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
        package_content["lake_tables"] = join_lake_tables(lake_numbers, lakes)
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
    
    def _has_tables(self):
        # item() will not work here if the object is an array.
        # .values[()] will simply return the full numpy array.
        tables = self.dataset["lake_tables"].values[()]
        if any([ pd.api.types.is_numeric_dtype(t) for t in tables  ]):
            return True
        return False    

    def _has_timeseries(self):
        for name in self._period_data:
            if "time" in self.dataset[name].coords:
                if len(self.dataset[name].coords["time"]) > 0:
                    return True
        return False

    @classmethod
    def _convert_to_string_dataarray(cls, x: xr.DataArray) -> xr.DataArray:
        # when adding a string dataarray to a dataset with more coordinates, the
        # values for coordinates in the dataset that are not present in the dataarray
        # are set to NaN, and the dataarray type changes to obj (because it now has both strings and NaNs)
        # This function can be used to convert such a dataarray back to string type. to detect nan's we cannot use np.isnan because that works only on numeric types.
        # instead we use the property that any equality check is false for nan's

        idx = np.where(x != x)
        x[idx[0][:]] = ""
        return x.astype(np.str)

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
        d["noutlets"] = 0
        if self._has_outlets():
            d["noutlets"] = len(self.dataset["outlet_lakein"])


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

        return self._template.render(d)

    def _get_iconn(self, lake_numbers_per_connection):
        iconn = np.full_like(
            lake_numbers_per_connection, dtype=np.integer, fill_value=0
        )
        maxlake = lake_numbers_per_connection.max()
        connections_per_lake = np.zeros(maxlake + 1)
        for i in range(np.size(lake_numbers_per_connection)):
            lakeno = lake_numbers_per_connection[i]
            connections_per_lake[lakeno] += 1
            iconn[i] = connections_per_lake[lakeno]
        return iconn

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
        lake_numbers_per_connection = self.dataset["connection_lake_number"].values

        data_df["iconn"] = self._get_iconn(lake_numbers_per_connection)
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
            if self._has_tables():
                lake_number_to_filename = self._write_laketable_filelist_section(f)
                self._write_laketable_files(directory, lake_number_to_filename)

            if self._has_outlets():
                f.write("\n")
                self._write_table_section(f, self._outlet_dataframe(), "outlets")

            if self._has_timeseries():
                self._write_period_section(f, globaltimes)

            return

    def _write_period_section(self, f, globaltimes):
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

        period_data_list = []

        period_data_name_list = [tssname for tssname in self._period_data]
        timeseries_dataset = self.dataset[period_data_name_list]
        timeseries_times = self.dataset.coords["time"]
        iperiods = np.searchsorted(globaltimes, timeseries_times) + 1
        for iperiod, (_, period_data) in zip(
            iperiods, timeseries_dataset.groupby("time")
        ):
            period_data_list.append(Period_internal(iperiod))
            for tssname in self._period_data:
                if len(period_data[tssname].dims) > 0:
                    for index in period_data.coords["index"].values:
                        value = period_data[tssname].sel(index=index).values[()]
                        isvalid = False
                        if isinstance(value, str):
                            isvalid = value != ""
                        else:
                            isvalid = ~np.isnan(value)

                        if isvalid:
                            period_data_list[-1].nr_values += 1
                            period_data_list[-1].lake_or_outlet_number.append(index)
                            period_data_list[-1].series_name.append(tssname[3:])
                            period_data_list[-1].value.append(value)

        _template = jinja2.Template(
            textwrap.dedent(
                """
        {% if nperiod > 0 -%}
        {% for iperiod in range(0, nperiod) %}
        {% if periods[iperiod].nr_values > 0 -%}
        begin period {{periods[iperiod].period_number}}{% for ivalue in range(0, periods[iperiod].nr_values) %}
          {{periods[iperiod].lake_or_outlet_number[ivalue]}}  {{periods[iperiod].series_name[ivalue]}} {{periods[iperiod].value[ivalue]}}{% endfor %}
        end period
        {% endif -%}
        {% endfor -%}
        {% endif -%}"""
            )
        )

        d = {}
        d["nperiod"] = len(period_data_list)
        d["periods"] = period_data_list

        period_block = _template.render(d)
        f.write(period_block)

    def _package_data_to_sparse(self):
        return

    def fill_stress_perioddata(self):
        #this function is called from packagebase and should do nothing in this context
        return

    def write_perioddata(self, directory, pkgname, binary):
        #this function is called from packagebase and should do nothing in this context        
        return
    
    def _write_laketable_filelist_section(self, f,):

        lake_number_to_lake_table_filename = {}
        f.write("tables  \n") 
        for name, number in zip(
            self.dataset["lake_boundname"],
            self.dataset["lake_number"],
        ):
            lake_number = number.values[()]
            lake_name = name.values[()]

            if lake_number in self.dataset["lake_tables"].coords["lake_nr"].values:
                table_file = lake_name + ".ltbl"
                f.write(f"   {lake_number}  TAB6 FILEIN {table_file}\n")
                lake_number_to_lake_table_filename[lake_number] = table_file
        return lake_number_to_lake_table_filename


    def _write_laketable_files(self, directory,  lake_number_to_filename):
        for num, file in lake_number_to_filename.items():
            table = self.dataset["lake_tables"].sel({"lake_nr":num,})
            ncol = 3
            stage_col = table.sel({"column":"stage"})
            if "barea" in table.coords["column"]:
                ncol = 4
            nrow = stage_col.where(pd.api.types.is_numeric_dtype).count().values[()]

            fullpath_laketable = directory /file 
            with open(fullpath_laketable, "w") as table_file:
                table_file.write("BEGIN DIMENSIONS\n")
                table_file.write(f"NROW {nrow}\n")
                table_file.write(f"NCOL {ncol}\n")     
                table_file.write("END DIMENSIONS\n") 
                table_file.write("begin table\n") 

                table_dataframe = pd.DataFrame(table.transpose())
                string_table = table_dataframe.iloc[range(nrow), range(ncol)].to_string( header=False, index=False)
                table_file.write(string_table)
                table_file.write("end table\n")                 
                                     


    def _write_table_section(
        self, f, dataframe: pd.DataFrame, title: str, index: bool = False
    ) -> None:
        f.write(f"begin {title}\n")
        block = dataframe.to_csv(
            index=index,
            header=False,
            sep=" ",
            line_terminator="\n",
        )
        trimmedlines = [line.strip() for line in block.splitlines()]
        trimmedblock = "\n".join(map(str, trimmedlines)) + "\n"
        f.write(trimmedblock)
        f.write(f"end {title}\n")
        return
