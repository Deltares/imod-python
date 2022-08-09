from imod.mf6.pkgbase import Package, AdvancedBoundaryCondition


class Lake_internal:
    '''
    The Lake_internal class is used for rendering the lake package in jinja.
    There is no point in instantiating this class as a user.
    '''
    def __init__(self):
        self.number = 0
        self.boundname = ""
        self.starting_stage = 0
        self.lake_bed_elevation = 0


class Connection_internal:
    '''
    The Connection_internal class is used for rendering the lake package in jinja.
    There is no point in instantiating this class as a user.
    '''
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


class Outlet_internal:
    '''
    The Outlet_internal class is used for rendering the lake package in jinja.
    There is no point in instantiating this class as a user.
    '''
    def __init__(self):
        self.lakein = 0
        self.lakeout = 0
        self.couttype = 0
        self.invert = 0
        self.roughness = 0
        self.width = 0
        self.slope = 0


class Lake(AdvancedBoundaryCondition):
    '''

    '''
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
        '''
        This function fills structs representing lakes, connections and outlets for the purpose of rendering
        this package with a jinja template.
        '''
        lakelist = []
        nlakes = self.dataset["l_number"].size
        for i in range (0, nlakes):
            lake = Lake_internal()
            lake.boundname = self.dataset["l_boundname"].values[i]
            lake.lake_bed_elevation = self.dataset["l_bed_elevation"].values[i]
            lake.number = self.dataset["l_number"].values[i]
            lake.starting_stage = self.dataset["l_starting_stage"].values[i]
            lakelist.append(lake)

        connectionlist = []
        nconnect = self.dataset["c_lake_no"].size
        for i in range (0, nconnect):
            connection = Connection_internal()
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
                outlet = Outlet_internal()
                outlet.lakein = self.dataset["o_lakein"].values[i]
                outlet.lakeout = self.dataset["o_lakeout"].values[i]
                outlet.couttype = self.dataset["o_couttype"].values[i]
                outlet.invert = self.dataset["o_invert"].values[i]
                outlet.roughness = self.dataset["o_roughness"].values[i]
                outlet.width = self.dataset["o_width"].values[i]
                outlet.slope = self.dataset["o_slope"].values[i]
                outletlist.append(outlet)

        return lakelist, connectionlist, outletlist

