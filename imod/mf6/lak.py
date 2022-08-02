from imod.mf6.pkgbase import Package, AdvancedBoundaryCondition

'''
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


        # table data
                lakeno,   # nrtables. integer value that defines the lake number associated with the specified TABLES data on the line. LAKENO must be greater than zero and less than or equal to NLAKES. The program will terminate with an error if table information for a lake is specified more than once or the number of specified tables is less than NTABLES.
                filename, # nrtables. filename containing table

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
class LakeLake:
    def __init__(self):
        self.number = 0
        self.boundname = ""
        self.starting_stage = 0
        self.lake_bed_elevation = 0

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
'''
class LakeConnection:
    def __init__(self):
        self.lake_no = 0
        self.cell_id_row_or_index = 0
        self.cell_id_col = 0
        self.cell_id_layer = 0
        self.connection_type = ""
        self.bed_leak =0
        self.bottom_elevation = 0
        self.top_elevation = 0
        self.connection_width = 0
        self.connection_length = 0

'''
        # outlets
                lakein, #(nroutlets)
                lakeout, #(nroutlets)
                couttype,# "specfied", "manning", "weir"
                invert,
                roughness,
                width,
                slope,
'''
class LakeOutlet:
    def __init__(self):
        self.lakein = 0
        self.lakeout = 0
        self.couttype = 0
        self.invert = 0
        self.roughness = 0
        self.width = 0
        self.slope = 0

class Lake(AdvancedBoundaryCondition):
    _pkg_id = "lak"
    _template = Package._initialize_template(_pkg_id)
    _metadata_dict = {}


    def __init__(
        #lake
        self, l_number, l_starting_stage, l_bed_elevation, l_boundname,
        #connection
        c_lake_no, c_cell_id_row_or_index, c_cell_id_col, c_cell_id_layer,  c_type, c_bed_leak, c_bottom_elevation, c_top_elevation, c_width, c_length,
        #outlet
        o_lakein = None, o_lakeout= None, o_couttype= None, o_invert= None, o_roughness= None, o_width= None, o_slope= None,
        #options
        print_input=False,
        print_stage =False,
        print_flows=False,
        save_flows=False,
        stagefile =None,
        budgetfile=None,
        budgetcsvfile =None,
        package_convergence_filename =None,
        ts6_filename =None,
        time_conversion=None,
        length_conversion=None
     ):
        super().__init__(locals())
        self.dataset["l_boundname"] = l_boundname
        self.dataset["l_bed_elevation"] = l_bed_elevation
        self.dataset["l_number"] = l_number
        self.dataset["l_starting_stage"] = l_starting_stage

        self.dataset["c_lake_no"] = c_lake_no
        self.dataset["c_cell_id_row_or_index"] = c_cell_id_row_or_index
        self.dataset["c_cell_id_col"] = c_cell_id_col
        self.dataset["c_cell_id_layer"] = c_cell_id_layer
        self.dataset["c_type"] = c_type
        self.dataset["c_bed_leak"] = c_bed_leak
        self.dataset["c_bottom_elevation"] = c_bottom_elevation
        self.dataset["c_top_elevation"] = c_top_elevation
        self.dataset["c_width"] = c_width
        self.dataset["c_length"] = c_length

        if o_lakein is not None:
            self.dataset["o_lakein"] = o_lakein
            self.dataset["o_lakeout"] = o_lakeout
            self.dataset["o_couttype"] = o_couttype
            self.dataset["o_invert"] = o_invert
            self.dataset["o_roughness"] = o_roughness
            self.dataset["o_width"] = o_width
            self.dataset["o_slope"] = o_slope

        self.dataset["print_input"]=print_input
        self.dataset["print_stage"] =print_stage
        self.dataset["print_flows"]=print_flows
        self.dataset["save_flows"]=save_flows

        self.dataset["stagefile"] =stagefile
        self.dataset["budgetfile"]=budgetfile
        self.dataset["budgetcsvfile"] =budgetcsvfile
        self.dataset["package_convergence_filename"] =package_convergence_filename
        self.dataset["ts6_filename"] =ts6_filename
        self.dataset["time_conversion"]=time_conversion
        self.dataset["length_conversion"]=length_conversion
        self._pkgcheck()

    def _package_data_to_sparse(self):
        i = 0


    def render(self, directory, pkgname, globaltimes, binary):
        d = {}

        for var in ("print_input","print_stage", "print_flows", "save_flows"):
            value = self[var].values[()]
            if self._valid(value):
                d[var] = value

        for var in ("stagefile", "budgetfile","budgetcsvfile", "package_convergence_filename", "ts6_filename","time_conversion", "length_conversion" ):
            value = self[var].values[()]
            if self._valid(value):
                d[var] = value

        lakelist, connectionlist, outletlist = self.get_structures_from_arrays()
        d["lakes"]= lakelist
        d["connections"]= connectionlist
        d["outlets"]= outletlist
        d["nlakes"]= len(lakelist)
        d["nconnect"] = len(connectionlist)
        d["noutlet"] = len(outletlist)
        d["ntables"] = 0
        return self._template.render(d)

    def get_structures_from_arrays(self):
        lakelist = []
        nlakes = self.dataset["l_number"].size
        for i in range (0, nlakes):
            lake = LakeLake()
            lake.boundname = self.dataset["l_boundname"].values[i]
            lake.lake_bed_elevation = self.dataset["l_bed_elevation"].values[i]
            lake.number = self.dataset["l_number"].values[i]
            lake.starting_stage = self.dataset["l_starting_stage"].values[i]
            lakelist.append(lake)

        connectionlist = []
        nconnect = self.dataset["c_lake_no"].size
        for i in range (0, nconnect):
            connection = LakeConnection()
            connection.lake_no= self.dataset["c_lake_no"].values[i]
            connection.cell_id_row_or_index =  self.dataset["c_cell_id_row_or_index"].values[i]
            connection.cell_id_col=None
            if self.dataset["c_cell_id_col"].values[()] is not None :
                connection.cell_id_col=  self.dataset["c_cell_id_col"].values[i]
            connection.cell_id_layer=  self.dataset["c_cell_id_layer"].values[i]

            connection.connection_type = self.dataset["c_type"].values[i]
            connection.bed_leak = self.dataset["c_bed_leak"].values[i]
            connection.bottom_elevation = self.dataset["c_bottom_elevation"].values[i]
            connection.top_elevation = self.dataset["c_top_elevation"].values[i]
            connection.connection_width = self.dataset["c_width"].values[i]
            connection.connection_length = self.dataset["c_length"].values[i]
            connectionlist.append(connection)

        outletlist = []
        if ("o_lakein" in  self.dataset.keys()):
            noutlet = self.dataset["o_lakein"].size
            for i in range(0, noutlet):
                outlet = LakeOutlet()
                outlet.lakein = self.dataset["o_lakein"].values[i]
                outlet.lakeout = self.dataset["o_lakeout"].values[i]
                outlet.couttype = self.dataset["o_couttype"].values[i]
                outlet.invert = self.dataset["o_invert"].values[i]
                outlet.roughness = self.dataset["o_roughness"].values[i]
                outlet.width = self.dataset["o_width"].values[i]
                outlet.slope = self.dataset["o_slope"].values[i]
                outletlist.append(outlet)

        return lakelist, connectionlist, outletlist

