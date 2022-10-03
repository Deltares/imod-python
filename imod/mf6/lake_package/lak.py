import collections

import numpy as np

from imod.mf6.pkgbase import BoundaryCondition, Package, VariableMetaData




class Lake(BoundaryCondition):
    """
    Lake (LAK) Package

    Parameters
    ----------
    l_number: array of integers (xr.DataArray)- dimension number of lakes:
        integer used as identifier for the lake.
    l_starting_stage: array of floats (xr.DataArray)- dimension number of lakes:
        starting lake stage.
    l_boundname:  array of strings (xr.DataArray)- dimension number of lakes:
        name of the lake

    c_lake_no: array of floats (xr.DataArray)- dimension number of connections
        lake number for the current lake-to-aquifer connection.
    c_cell_id_row_or_index: array of floats (xr.DataArray)- dimension number of connections
        in case of a structured grid: gridrow number of aquifer cell for current lake-to aquifer connection
        in case of an unstructured grid: cell index of aquifer cell for current lake-to aquifer connection

    c_cell_id_col: array of floats (xr.DataArray)- dimension number of connections
        in case of a structured grid: grid-column number of aquifer cell for current lake-to aquifer connection
        in case of an unstructured grid: not used. set this argument to None

    c_cell_id_layer: array of floats (xr.DataArray)- dimension number of connections
         gridrow number of aquifer cell for current lake-to aquifer connection
    c_type: array of strings (xr.DataArray)- dimension number of connections
        indicates if connection is horizontal, vertical, embeddedH or embeddedV
    c_bed_leak: array of floats (xr.DataArray)- dimension number of connections
        defines the bed leakance for the lake-GWF connection.
        BEDLEAK must be greater than or equal to zero or specified to be np.nan. If BEDLEAK is specified to
        be np.nan, the lake-GWF connection conductance is solely a function of aquifer properties in the
        connected GWF cell and lakebed sediments are assumed to be absent.
    c_bottom_elevation: array of floats (xr.DataArray, optional)- dimension number of connections
        defines the bottom elevation for a horizontal lake-GWF connection.
        If not provided, will be set to the bottom elevation of the cell it is connected to.
    c_top_elevation:array of floats (xr.DataArray, optional)- dimension number of connections
        defines the top elevation for a horizontal lake-GWF connection.
        If not provided, will be set to the top elevation of the cell it is connected to.
    c_width: array of floats (xr.DataArray, optional)
        defines the connection face width for a horizontal lake-GWF connection.
        connwidth must be greater than zero for a horizontal lake-GWF connection. Any value can be
        specified if claktype is vertical, embeddedh, or embeddedv. If not set, will be set to dx or dy.
    c_length: array of floats (xr.DataArray, optional)
        defines the distance between the connected GWF cellid node and the lake
        for a horizontal, embeddedh, or embeddedv lake-GWF connection. connlen must be greater than
        zero for a horizontal, embeddedh, or embeddedv lake-GWF connection. Any value can be specified
        if claktype is vertical. If not set, will be set to dx or dy.


    o_lakein: array of integers (xr.DataArray, optional)
        integer defining the lake number that outlet is connected to. Must be
        greater than zero.
    o_lakeout: array of integers (xr.DataArray, optional)
         integer value that defines the lake number that outlet discharge from lake outlet OUTLETNO
        is routed to. Must be greater than or equal to zero.
        If zero, outlet discharge from lake outlet OUTLETNO is discharged to an external
        boundary.
    o_couttype: array of string (xr.DataArray, optional)
        character string that defines the outlet type for the outlet OUTLETNO. Possible
        strings include: SPECIFIED–character keyword to indicate the outlet is defined as a specified
        flow. MANNING–character keyword to indicate the outlet is defined using Manning’s equation.
        WEIR–character keyword to indicate the outlet is defined using a sharp weir equation.
    o_invert: array of floats (xr.DataArray, optional):
        float or character value that defines the invert elevation for the lake outlet. A specified
        INVERT value is only used for active lakes if outlet_type for lake outlet OUTLETNO is not
        SPECIFIED.
    o_roughness: array of floats (xr.DataArray, optional)
        defines the roughness coefficient for the lake outlet. Any value can be specified
        if outlet_type is not MANNING.
    o_width: array of floats (xr.DataArray, optional)
        float or character value that defines the width of the lake outlet. A specified WIDTH value is
        only used for active lakes if outlet_type for lake outlet OUTLETNO is not SPECIFIED.
    o_slope: array of floats (xr.DataArray, optional)
        float or character value that defines the bed slope for the lake outlet. A specified SLOPE value is
        only used for active lakes if outlet_type for lake outlet OUTLETNO is MANNING.

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
        "l_number": VariableMetaData(np.integer),
        "l_starting_stage": VariableMetaData(np.floating),
        "l_boundname": VariableMetaData(np.str0),
        "c_lake_no": VariableMetaData(np.floating),
        "c_cell_id_row_or_index": VariableMetaData(np.floating),
        "c_cell_id_col": VariableMetaData(np.floating),
        "c_cell_id_layer": VariableMetaData(np.floating),
        "c_type": VariableMetaData(np.floating),
        "c_bed_leak": VariableMetaData(np.floating),
        "c_bottom_elevation": VariableMetaData(np.floating),
        "c_top_elevation": VariableMetaData(np.floating),
        "c_width": VariableMetaData(np.floating),
        "c_length": VariableMetaData(np.floating),
        "o_lakein": VariableMetaData(np.integer),
        "o_lakeout": VariableMetaData(np.integer),
        "o_couttype": VariableMetaData(np.str0),
        "o_invert": VariableMetaData(np.floating),
        "o_roughness": VariableMetaData(np.floating),
        "o_width": VariableMetaData(np.floating),
        "o_slope": VariableMetaData(np.floating),
    }

    connection_types = {"horizontal": 0, "vertical": 1, "embeddedh": 2, "embeddedv": 3}
    inverse_connection_types = {v: k for k, v in connection_types.items()}

    def __init__(
        # lake
        self,
        l_number,
        l_starting_stage,
        l_boundname,
        # connection
        c_lake_no,
        c_cell_id_row_or_index,
        c_cell_id_col,
        c_cell_id_layer,
        c_type,
        c_bed_leak,
        c_bottom_elevation,
        c_top_elevation,
        c_width,
        c_length,
        # outlet
        o_lakein=None,
        o_lakeout=None,
        o_couttype=None,
        o_invert=None,
        o_roughness=None,
        o_width=None,
        o_slope=None,
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
        self.dataset["l_boundname"] = l_boundname
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

        self.dataset["o_lakein"] = o_lakein
        self.dataset["o_lakeout"] = o_lakeout
        self.dataset["o_couttype"] = o_couttype
        self.dataset["o_invert"] = o_invert
        self.dataset["o_roughness"] = o_roughness
        self.dataset["o_width"] = o_width
        self.dataset["o_slope"] = o_slope

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
        self._pkgcheck()

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}

        for var in ("print_input", "print_stage", "print_flows", "save_flows"):
            value = self[var].values[()]
            if self._valid(value):
                d[var] = value

        for var in (
            "stagefile",
            "budgetfile",
            "budgetcsvfile",
            "package_convergence_filename",
            "ts6_filename",
            "time_conversion",
            "length_conversion",
        ):
            value = self[var].values[()]
            if self._valid(value):
                d[var] = value

        lakelist, connectionlist, outletlist = self.get_structures_from_arrays()
        d["lakes"] = lakelist
        d["connections"] = connectionlist
        d["outlets"] = outletlist
        d["nlakes"] = len(lakelist.boundname)
        d["nconnect"] = len(connectionlist.lake_no)
        d["noutlets"] = 0
        if outletlist is not None:
            d["noutlets"] = len(outletlist.lake_in)
        d["ntables"] = 0
        return self._template.render(d)

    def get_structures_from_arrays(self):
        """
        This function fills structs representing lakes, connections and outlets for the purpose of rendering
        this package with a jinja template.
        """
        nlakes = len(self.dataset["l_boundname"].values)
        LakeList = collections.namedtuple(
            "lakelist", ["number", "boundname", "starting_stage", "nconn"]
        )
        nconn = [0] * nlakes
        lakelist = LakeList(
            self.dataset["l_number"].values,
            self.dataset["l_boundname"].values,
            self.dataset["l_starting_stage"].values,
            nconn,
        )

        ConnectionList = collections.namedtuple(
            "connectionlist",
            [
                "lake_no",
                "connection_nr",
                "cell_id_row_or_index",
                "cell_id_col",
                "cell_id_layer",
                "connection_type",
                "bed_leak",
                "bottom_elevation",
                "top_elevation",
                "connection_width",
                "connection_length",
            ],
        )
        clakeno = self.dataset["c_lake_no"].values.astype(int)
        conn_number = [0] * len(clakeno)
        ccellid = self.dataset["c_cell_id_row_or_index"].values.astype(int)
        ccelcol = None
        if self.dataset["c_cell_id_col"].values[()] is not None:
            ccelcol = self.dataset["c_cell_id_col"].values.astype(int)
        ccellayer = self.dataset["c_cell_id_layer"].values.astype(int)

        cbed_leak = self.dataset["c_bed_leak"].values
        cbottom_elevation = self.dataset["c_bottom_elevation"].values
        ctop_elevation = self.dataset["c_top_elevation"].values
        cwidth = self.dataset["c_width"].values
        clength = self.dataset["c_length"].values
        ctype = np.vectorize(self.inverse_connection_types.get)(
            self.dataset["c_type"].values
        )
        connectionlist = ConnectionList(
            clakeno,
            conn_number,
            ccellid,
            ccelcol,
            ccellayer,
            ctype,
            cbed_leak,
            cbottom_elevation,
            ctop_elevation,
            cwidth,
            clength,
        )

        self.fill_in_connection_number(lakelist, connectionlist)

        OutletList = collections.namedtuple(
            "outletlist",
            [
                "lake_in",
                "lake_out",
                "couttype",
                "invert",
                "roughness",
                "width",
                "slope",
            ],
        )
        clakeno = self.dataset["c_lake_no"].values.astype(int)
        outletlist = None
        if self._valid(self.dataset["o_lakein"].values[()]):
            outletlist = OutletList(
                self.dataset["o_lakein"].values.astype(int),
                self.dataset["o_lakeout"].values.astype(int),
                self.dataset["o_couttype"].values,
                self.dataset["o_invert"].values,
                self.dataset["o_roughness"].values,
                self.dataset["o_width"].values,
                self.dataset["o_slope"].values,
            )

        return lakelist, connectionlist, outletlist

    def fill_in_connection_number(self, lakelist, connectionlist):

        # count connections per lake. FIll in the nconn attribute of the lake.
        # also fill in the connection number (of each connection in each lake)
        connections_per_lake = lakelist.nconn
        nr_lakes = len(connections_per_lake)

        connection_numbers = connectionlist.connection_nr
        nr_connections = len(connection_numbers)
        connection_lake_number = connectionlist.lake_no

        for i in range(0, nr_lakes):
            connections_per_lake[i] = 0

        for iconnection in range(0, nr_connections):
            lakenumber = connection_lake_number[iconnection]
            connections_per_lake[lakenumber - 1] += 1
            connection_numbers[iconnection] = connections_per_lake[lakenumber - 1]

        return

    def _package_data_to_sparse(self):
        return

    def fill_stress_perioddata(self):
        return
