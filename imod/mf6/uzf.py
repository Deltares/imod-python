import numpy as np
import xarray as xr
import xugrid as xu

from imod.logging import init_log_decorator
from imod.mf6.boundary_condition import AdvancedBoundaryCondition, BoundaryCondition
from imod.mf6.validation import BOUNDARY_DIMS_SCHEMA
from imod.schemata import (
    AllCoordsValueSchema,
    AllInsideNoDataSchema,
    AllNoDataSchema,
    AllValueSchema,
    CoordsSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
    OtherCoordsSchema,
)
from imod.select.layers import get_upper_active_grid_cells


class UnsaturatedZoneFlow(AdvancedBoundaryCondition):
    """
    Unsaturated Zone Flow (UZF) package.

    Parameters
    ----------
    surface_depression_depth: array of floats (xr.DataArray)
        is the surface depression depth of the UZF cell.
    kv_sat: array of floats (xr.DataArray)
        is the vertical saturated hydraulic conductivity of the UZF cell.
        NOTE: the UZF package determines the location of inactive cells where kv_sat is np.nan
    theta_res: array of floats (xr.DataArray)
        is the residual (irreducible) water content of the UZF cell.
    theta_sat: array of floats (xr.DataArray)
        is the saturated water content of the UZF cell.
    theta_init: array of floats (xr.DataArray)
        is the initial water content of the UZF cell.
    epsilon: array of floats (xr.DataArray)
        is the epsilon exponent of the UZF cell.
    infiltration_rate: array of floats (xr.DataArray)
        defines the applied infiltration rate of the UZF cell (LT -1).
    et_pot: array of floats (xr.DataArray, optional)
        defines the potential evapotranspiration rate of the UZF cell and specified
        GWF cell. Evapotranspiration is first removed from the unsaturated zone and any remaining
        potential evapotranspiration is applied to the saturated zone. If IVERTCON is greater than zero
        then residual potential evapotranspiration not satisfied in the UZF cell is applied to the underlying
        UZF and GWF cells.
    extinction_depth: array of floats (xr.DataArray, optional)
        defines the evapotranspiration extinction depth of the UZF cell. If
        IVERTCON is greater than zero and EXTDP extends below the GWF cell bottom then remaining
        potential evapotranspiration is applied to the underlying UZF and GWF cells. EXTDP is always
        specified, but is only used if SIMULATE ET is specified in the OPTIONS block.
    extinction_theta: array of floats (xr.DataArray, optional)
        defines the evapotranspiration extinction water content of the UZF
        cell. If specified, ET in the unsaturated zone will be simulated either as a function of the
        specified PET rate while the water content (THETA) is greater than the ET extinction water content
    air_entry_potential: array of floats (xr.DataArray, optional)
        defines the air entry potential (head) of the UZF cell. If specified, ET will be
        simulated using a capillary pressure based formulation.
        Capillary pressure is calculated using the Brooks-Corey retention function ("air_entry")
    root_potential: array of floats (xr.DataArray, optional)
        defines the root potential (head) of the UZF cell. If specified, ET will be
        simulated using a capillary pressure based formulation.
        Capillary pressure is calculated using the Brooks-Corey retention function ("air_entry"
    root_activity: array of floats (xr.DataArray, optional)
        defines the root activity function of the UZF cell. ROOTACT is
        the length of roots in a given volume of soil divided by that volume. Values range from 0 to about 3
        cm-2, depending on the plant community and its stage of development. If specified, ET will be
        simulated using a capillary pressure based formulation.
        Capillary pressure is calculated using the Brooks-Corey retention function ("air_entry"
    groundwater_ET_function: ({"linear", "square"}, optional)
        keyword specifying that groundwater evapotranspiration will be simulated using either
        the original ET formulation of MODFLOW-2005 ("linear"). Or by assuming a constant ET
        rate for groundwater levels between land surface (TOP) and land surface minus the ET extinction
        depth (TOP-EXTDP) ("square"). In the latter case, groundwater ET is smoothly reduced
        from the PET rate to zero over a nominal interval at TOP-EXTDP.
    simulate_seepage: ({True, False}, optional)
        keyword specifying that groundwater discharge (GWSEEP) to land surface will be
        simulated. Groundwater discharge is nonzero when groundwater head is greater than land surface.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of UZF information will be written to the listing file
        immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        keyword to indicate that the list of UZF flow rates will be printed to the listing file for
        every stress period time step in which "BUDGET PRINT" is specified in Output Control. If there is
        no Output Control option and "PRINT FLOWS" is specified, then flow rates are printed for the last
        time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        keyword to indicate that UZF flow terms will be written to the file specified with "BUDGET
        FILEOUT" in Output Control.
        Default is False.
    budget_fileout: ({"str"}, optional)
        path to output cbc-file for UZF budgets
    budgetcsv_fileout: ({"str"}, optional)
        path to output csv-file for summed budgets
    water_content_file: ({"str"}, optional)
        path to output file for unsaturated zone water content
    observations: [Not yet supported.]
        Default is None.
    water_mover: [Not yet supported.]
        Default is None.
    timeseries: [Not yet supported.]
        Default is None.
        TODO: We could allow the user to either use xarray DataArrays to specify BCS or
        use a pd.DataFrame and use the MF6 timeseries files to read input. The latter could
        save memory for laterally large-scale models, through efficient use of the UZF cell identifiers.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _period_data = (
        "infiltration_rate",
        "et_pot",
        "extinction_depth",
        "extinction_theta",
        "air_entry_potential",
        "root_potential",
        "root_activity",
    )

    _init_schemata = {
        "surface_depression_depth": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "kv_sat": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BOUNDARY_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "theta_res": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "theta_sat": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "theta_init": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "epsilon": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "infiltration_rate": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "et_pot": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA | DimsSchema(),  # optional var
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "extinction_depth": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA | DimsSchema(),  # optional var
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "extinction_theta": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA | DimsSchema(),  # optional var
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "root_potential": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA | DimsSchema(),  # optional var
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "root_activity": [
            DTypeSchema(np.floating),
            BOUNDARY_DIMS_SCHEMA | DimsSchema(),  # optional var
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "print_flows": [DTypeSchema(np.bool_), DimsSchema()],
        "save_flows": [DTypeSchema(np.bool_), DimsSchema()],
    }
    _write_schemata = {
        "kv_sat": [
            AllValueSchema(">", 0.0),
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "surface_depression_depth": [IdentityNoDataSchema("kv_sat")],
        "theta_res": [IdentityNoDataSchema("kv_sat"), AllValueSchema(">=", 0.0)],
        "theta_sat": [IdentityNoDataSchema("kv_sat"), AllValueSchema(">=", 0.0)],
        "theta_init": [IdentityNoDataSchema("kv_sat"), AllValueSchema(">=", 0.0)],
        "epsilon": [IdentityNoDataSchema("kv_sat")],
        "infiltration_rate": [IdentityNoDataSchema("stress_period_active")],
        "et_pot": [IdentityNoDataSchema("stress_period_active")],
        "extinction_depth": [IdentityNoDataSchema("stress_period_active")],
        "extinction_theta": [IdentityNoDataSchema("stress_period_active")],
        "root_potential": [IdentityNoDataSchema("stress_period_active")],
        "root_activity": [IdentityNoDataSchema("stress_period_active")],
    }

    _package_data = (
        "surface_depression_depth",
        "kv_sat",
        "theta_res",
        "theta_sat",
        "theta_init",
        "epsilon",
    )
    _pkg_id = "uzf"

    _template = BoundaryCondition._initialize_template(_pkg_id)

    @init_log_decorator()
    def __init__(
        self,
        surface_depression_depth,
        kv_sat,
        theta_res,
        theta_sat,
        theta_init,
        epsilon,
        infiltration_rate,
        et_pot=None,
        extinction_depth=None,
        extinction_theta=None,
        air_entry_potential=None,
        root_potential=None,
        root_activity=None,
        ntrailwaves=7,  # Recommended in manual
        nwavesets=40,
        groundwater_ET_function=None,
        simulate_groundwater_seepage=False,
        print_input=False,
        print_flows=False,
        save_flows=False,
        budget_fileout=None,
        budgetcsv_fileout=None,
        water_content_file=None,
        observations=None,
        water_mover=None,
        timeseries=None,
        validate: bool = True,
    ):
        landflag = self._determine_landflag(kv_sat)
        iuzno = self._create_uzf_numbers(landflag)
        ivertcon = self._determine_vertical_connection(iuzno)
        stress_period_active = landflag.where(landflag == 1)

        dict_dataset = {
            # Package data
            "surface_depression_depth": surface_depression_depth,
            "kv_sat": kv_sat,
            "theta_res": theta_res,
            "theta_sat": theta_sat,
            "theta_init": theta_init,
            "epsilon": epsilon,
            # Stress period data
            "stress_period_active": stress_period_active,
            "infiltration_rate": infiltration_rate,
            "et_pot": et_pot,
            "extinction_depth": extinction_depth,
            "extinction_theta": extinction_theta,
            "air_entry_potential": air_entry_potential,
            "root_potential": root_potential,
            "root_activity": root_activity,
            # Dimensions
            "ntrailwaves": ntrailwaves,
            "nwavesets": nwavesets,
            # Options
            "groundwater_ET_function": groundwater_ET_function,
            "simulate_gwseep": simulate_groundwater_seepage,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "budget_fileout": budget_fileout,
            "budgetcsv_fileout": budgetcsv_fileout,
            "water_content_file": water_content_file,
            "observations": observations,
            "water_mover": water_mover,
            "timeseries": timeseries,
            # Additonal indices for Packagedata
            "landflag": landflag,
            "iuzno": iuzno,
            "ivertcon": ivertcon,
        }
        super().__init__(dict_dataset)
        self.dataset["iuzno"].name = "uzf_number"
        self._check_options(
            groundwater_ET_function,
            et_pot,
            extinction_depth,
            extinction_theta,
            air_entry_potential,
            root_potential,
            root_activity,
        )
        self._validate_init_schemata(validate)

    def _fill_stress_perioddata(self):
        """Modflow6 requires something to be filled in the stress perioddata,
        even though the data is not used in the current configuration.
        Only an infiltration rate is required,
        the rest can be filled with dummy values if not provided.
        """
        for var in self._period_data:
            if self.dataset[var].size == 1:  # Prevent loading large arrays in memory
                if self.dataset[var].values[()] is None:
                    if isinstance(self["infiltration_rate"], xu.UgridDataArray):
                        self.dataset[var] = xu.full_like(self["infiltration_rate"], 0)
                    else:
                        self.dataset[var] = xr.full_like(self["infiltration_rate"], 0.0)
                else:
                    raise ValueError("{} cannot be a scalar".format(var))

    def _check_options(
        self,
        groundwater_ET_function,
        et_pot,
        extinction_depth,
        extinction_theta,
        air_entry_potential,
        root_potential,
        root_activity,
    ):
        simulate_et = [x is not None for x in [et_pot, extinction_depth]]
        unsat_etae = [
            x is not None for x in [air_entry_potential, root_potential, root_activity]
        ]

        if all(simulate_et):
            self.dataset["simulate_et"] = True
        elif any(simulate_et):
            raise ValueError("To simulate ET, set both et_pot and extinction_depth")

        if extinction_theta is not None:
            self.dataset["unsat_etwc"] = True

        if all(unsat_etae):
            self.dataset["unsat_etae"] = True
        elif any(unsat_etae):
            raise ValueError(
                "To simulate ET with a capillary based formulation, set air_entry_potential, root_potential, and root_activity"
            )

        if all(unsat_etae) and (extinction_theta is not None):
            raise ValueError(
                """Both capillary based formulation and water content based formulation set based on provided input data.
                Please provide either only extinction_theta or (air_entry_potential, root_potential, and root_activity)"""
            )

        if groundwater_ET_function == "linear":
            self.dataset["linear_gwet"] = True
        elif groundwater_ET_function == "square":
            self.dataset["square_gwet"] = True
        elif groundwater_ET_function is None:
            pass
        else:
            raise ValueError(
                "Groundwater ET function should be either 'linear','square' or None"
            )

    def _create_uzf_numbers(self, landflag):
        """Create unique UZF ID's. Inactive cells equal 0"""
        active_nodes = landflag.notnull().astype(np.int8)
        return np.nancumsum(active_nodes).reshape(landflag.shape) * active_nodes

    def _determine_landflag(self, kv_sat):
        """returns the landflag for uzf-model. Landflag == 1 for top active UZF-nodes"""
        land_nodes = get_upper_active_grid_cells(kv_sat).astype(np.int32)
        return land_nodes.where(kv_sat.notnull())

    def _determine_vertical_connection(self, uzf_number):
        return uzf_number.shift(layer=-1, fill_value=0)

    def _package_data_to_sparse(self):
        notnull = self.dataset["landflag"].notnull().to_numpy()
        iuzno = self.dataset["iuzno"].values[notnull]
        landflag = self.dataset["landflag"].values[notnull]
        ivertcon = self.dataset["ivertcon"].values[notnull]

        ds = self.dataset[list(self._package_data)]

        layer = ds["layer"].values
        arrdict = self._ds_to_arrdict(ds)
        recarr = super()._to_struct_array(arrdict, layer)

        field_spec = self._get_field_spec_from_dtype(recarr)
        field_names = [i[0] for i in field_spec]
        n = 3
        if isinstance(self.dataset, xu.UgridDataset):
            n = 2
        index_spec = [("iuzno", np.int32)] + field_spec[:n]
        field_spec = (
            [("landflag", np.int32)] + [("ivertcon", np.int32)] + field_spec[n:]
        )
        sparse_dtype = np.dtype(index_spec + field_spec)

        recarr_new = np.empty(recarr.shape, dtype=sparse_dtype)
        recarr_new["iuzno"] = iuzno
        recarr_new["landflag"] = landflag
        recarr_new["ivertcon"] = ivertcon
        recarr_new[field_names] = recarr

        return recarr_new

    def _render(self, directory, pkgname, globaltimes, binary):
        """Render fills in the template only, doesn't write binary data"""
        d = {}
        bin_ds = self.dataset[list(self._period_data)]
        d["periods"] = self._period_paths(
            directory, pkgname, globaltimes, bin_ds, binary=False
        )
        not_options = (
            list(self._period_data) + list(self._package_data) + ["iuzno" + "ivertcon"]
        )
        d = self._get_pkg_options(d, not_options=not_options)
        path = directory / pkgname / f"{self._pkg_id}-pkgdata.dat"
        d["packagedata"] = path.as_posix()
        # max uzf-cells for which time period data will be supplied
        d["nuzfcells"] = np.count_nonzero(np.isfinite(d["landflag"]))
        return self._template.render(d)

    def _to_struct_array(self, arrdict, layer):
        """Convert from dense arrays to list based input,
        since the perioddata does not require cellids but iuzno, we hgave to override"""
        # TODO add pkgcheck that period table aligns
        # Get the number of valid values
        notnull = self.dataset["landflag"].values == 1
        iuzno = self.dataset["iuzno"].values
        nrow = notnull.sum()
        # Define the numpy structured array dtype
        index_spec = [("iuzno", np.int32)]
        field_spec = [(key, np.float64) for key in arrdict]
        sparse_dtype = np.dtype(index_spec + field_spec)

        # Initialize the structured array
        recarr = np.empty(nrow, dtype=sparse_dtype)
        # Fill in the indices
        recarr["iuzno"] = iuzno[notnull]

        # Fill in the data
        for key, arr in arrdict.items():
            recarr[key] = arr[notnull].astype(np.float64)

        return recarr

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["kv_sat"] = self["kv_sat"]
        kwargs["stress_period_active"] = self["stress_period_active"]
        errors = super()._validate(schemata, **kwargs)

        return errors
