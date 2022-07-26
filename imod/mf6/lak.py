import numpy as np

from imod.mf6.pkgbase import Package, AdvancedBoundaryCondition
class LakeLake:
    def __init__(self):
        self.number = 0
        self.boundname = ""
        self.starting_stage = 0
        self.lake_bed_elevation = 0
"""
    Parameters
    ----------
    lake_number: xr.DataArray of floats
        dims: (layer, y, x)
    lake_type: xr.DataArray of floats
        dims: (y, x)
    bed_leak: xr.DataArray of floats
        dims: (y, x)
    bottom_elevation: xr.DataArray of floats
        dims: (y, x)
    top_elevation: xr.DataArray of floats
        dims: (y, x)
    connection_width: xr.DataArray of floats
        dims: (y, x)
    connection_length: xr.DataArray of floats
        dims: (y, x)
    stage: xr.DataArray of floats
        dims: lake, n_segment
    volume: xr.DataArray of floats
        dims: lake, n_segment
    surface_area: xr.DataArray of floats
        dims: lake, n_segment
    bed_area: xr.DataArray of floats
        dims: lake, n_segment

    lakeout: xr.DataArray of integers
        Per outlet the lake_number it belongs to
        dims: (outlet,)
    lakein:
        dims: (outlet,)
    outlet_type: xr.Datarray of strings
        Values: manning, weir, specified
        dims: (outlet,)
    width: xr.DataArray of floats
        dims: (outlet,)
    roughness: xr.DataArray of floats
        dims: (outlet,)
    slope: xr.DataArray of floats
        dims: (outlet,)

    connections:
        dims: (outlet, 2); (outlet, (lake_out, lake_in))

    stage: xr.DataArray of floats
        dims: (time, lake)
    rainfall: xr.DataArray of floats
        dims: (time, lake)
    evaporation: xr.DataArray of floats
        dims: (time, lake)
    runoff: xr.DataArray of floats
        dims: (time, lake)
    inflow: xr.DataArray of floats
        dims: (time, lake)
    withdrawal: xr.DataArray of floats
        dims: (time, lake)
    rate: xr.DataArray of floats
        dims: (time, lake)
    invert: xr.DataArray of floats
        dims: (time, lake)


    roughness: xr.DataArray of floats
        dims: (time, lake)
    width: xr.DataArray of floats
        dims: (time, lake)
    slope: xr.DataArray of floats
        dims: (time, lake)

    """
class Lake(AdvancedBoundaryCondition):
    _pkg_id = "lak"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self,lake_number, starting_stage,bed_elevation,boundname ):

        #dimensions:
        #   implicit
        #package data: dimension nr of lakes
        #nlakeconn : number of connections of this lake. we should be able to compute this from the
        #connection data specified below
        #aux: the auxiliary variables for this lake. These should exist if we solve with transport.
        #name of the lake

        '''
        # connection data
        lake_no,  # either an array of n(i) cellindices for i lakes, or an array of size idomain with a lake number or a NaN in each cell
        cell_id,  # cell indices- either row, col, lay; or cell id depending on discretization
        connection_type, # = claktype. Tye of connection. "horizontal", "vertical", "embedded"
        bed_leak,  # >= s0 or None
        bottom_elevation,  #ony used for horizontal connections
        top_elevation,  #ony used for horizontal connections
        connection_width, #only for  HORIZONTAL, EMBEDDEDH, or EMBEDDEDV
        connection_length, #only for  HORIZONTAL, EMBEDDEDH, or EMBEDDEDV


        # table data
                lakeno,   # nrtables. integer value that defines the lake number associated with the specified TABLES data on the line. LAKENO must be greater than zero and less than or equal to NLAKES. The program will terminate with an error if table information for a lake is specified more than once or the number of specified tables is less than NTABLES.
                filename, # nrtables. filename containing table
        # outlets
                lakein, #(nroutlets)
                lakeout, #(nroutlets)
                couttype,# "specfied", "manning", "weir"
                invert,
                roughness,
                width,
                slope,
        #time series
                iper, #index of starting stress period
                number_ts, #index of lake or outlet
                laksetting_ts,# indicates what timeseries is for. 'STATUS, STAGE, RAINFALL, EVAPORATION, RUNOFF, INFLOW, WITHDRAWAL, and AUXILIARY for lake, TATUS, STAGE, RAINFALL, EVAPORATION, RUNOFF, INFLOW, WITHDRAWAL, and AUXILIARY for outlet
                status_ts, #
                stage_ts,
                rainfall_ts,
                evaporation_ts,
                runoff_ts,
                inflow_ts,
                withdrawal_ts,
                rate_ts,
                invert_ts,
                rough_ts,
                width_ts,
                slope_ts,
                auxname_ts,
                auxvalue_ts

        #options
                print_input=False,
                PRINT_STAGE =False,
                print_flows=False,
                save_flows=False,
                observations=None,
                stagefile =None,
                budgetfile=None,
                budgetcsvfile =None,
                package_convergence_filename =None,
                ts6_filename =None,
                time_conversion=None,
                length_conversion=None
        '''
        nlakes = lake_number.size
        lakelist = []
        for i in range (0, nlakes):
            lake = LakeLake()
            lake.boundname = boundname[i]
            lake.lake_bed_elevation = bed_elevation[i]
            lake.number = lake_number[i]
            lake.starting_stage = starting_stage[i]
            lakelist.append(lake)


    def _package_data_to_sparse(self):
        i = 0