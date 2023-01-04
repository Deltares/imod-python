import numpy as np
import xarray as xr

from imod.mf6.pkgbase import AdvancedBoundaryCondition, BoundaryCondition
from imod.mf6.validation import BC_DIMS_SCHEMA
from imod.schemata import (
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


class UnsaturatedZoneFlow(AdvancedBoundaryCondition):
    """
    Unsaturated Zone Flow (UZF) package.

    TODO: Support timeseries file? Observations? Water Mover?

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
            BC_DIMS_SCHEMA,
        ],
        "kv_sat": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            BC_DIMS_SCHEMA,
        ],
        "theta_res": [
            DTypeSchema(np.floating),
            BC_DIMS_SCHEMA,
        ],
        "theta_sat": [
            DTypeSchema(np.floating),
            BC_DIMS_SCHEMA,
        ],
        "theta_init": [
            DTypeSchema(np.floating),
            BC_DIMS_SCHEMA,
        ],
        "epsilon": [
            DTypeSchema(np.floating),
            BC_DIMS_SCHEMA,
        ],
        "infiltration_rate": [
            DTypeSchema(np.floating),
            BC_DIMS_SCHEMA,
        ],
        "et_pot": [
            DTypeSchema(np.floating),
            BC_DIMS_SCHEMA | DimsSchema(),  # optional var
        ],
        "extinction_depth": [
            DTypeSchema(np.floating),
            BC_DIMS_SCHEMA | DimsSchema(),  # optional var
        ],
        "extinction_theta": [
            DTypeSchema(np.floating),
            BC_DIMS_SCHEMA | DimsSchema(),  # optional var
        ],
        "root_potential": [
            DTypeSchema(np.floating),
            BC_DIMS_SCHEMA | DimsSchema(),  # optional var
        ],
        "root_activity": [
            DTypeSchema(np.floating),
            BC_DIMS_SCHEMA | DimsSchema(),  # optional var
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
        "infiltration_rate": [IdentityNoDataSchema("kv_sat")],
        "et_pot": [IdentityNoDataSchema("kv_sat")],
        "extinction_depth": [IdentityNoDataSchema("kv_sat")],
        "extinction_theta": [IdentityNoDataSchema("kv_sat")],
        "root_potential": [IdentityNoDataSchema("kv_sat")],
        "root_activity": [IdentityNoDataSchema("kv_sat")],
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
        observations=None,
        water_mover=None,
        timeseries=None,
        validate=True,
    ):
        super().__init__(locals())
        # Package data
        self.dataset["surface_depression_depth"] = surface_depression_depth
        self.dataset["kv_sat"] = kv_sat
        self.dataset["theta_res"] = theta_res
        self.dataset["theta_sat"] = theta_sat
        self.dataset["theta_init"] = theta_init
        self.dataset["epsilon"] = epsilon

        # Stress period data
        self._check_options(
            groundwater_ET_function,
            et_pot,
            extinction_depth,
            extinction_theta,
            air_entry_potential,
            root_potential,
            root_activity,
        )

        self.dataset["infiltration_rate"] = infiltration_rate
        self.dataset["et_pot"] = et_pot
        self.dataset["extinction_depth"] = extinction_depth
        self.dataset["extinction_theta"] = extinction_theta
        self.dataset["air_entry_potential"] = air_entry_potential
        self.dataset["root_potential"] = root_potential
        self.dataset["root_activity"] = root_activity

        # Dimensions
        self.dataset["ntrailwaves"] = ntrailwaves
        self.dataset["nwavesets"] = nwavesets

        # Options
        self.dataset["groundwater_ET_function"] = groundwater_ET_function
        self.dataset["simulate_gwseep"] = simulate_groundwater_seepage
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations
        self.dataset["water_mover"] = water_mover
        self.dataset["timeseries"] = timeseries

        # Additonal indices for Packagedata
        self.dataset["landflag"] = self._determine_landflag(kv_sat)

        self.dataset["iuzno"] = self._create_uzf_numbers(self["landflag"])
        self.dataset["iuzno"].name = "uzf_number"

        self.dataset["ivertcon"] = self._determine_vertical_connection(self["iuzno"])

        if validate:
            self._validate_at_init()

    def fill_stress_perioddata(self):
        """Modflow6 requires something to be filled in the stress perioddata,
        even though the data is not used in the current configuration.
        Only an infiltration rate is required,
        the rest can be filled with dummy values if not provided.
        """
        for var in self._period_data:
            if self.dataset[var].size == 1:  # Prevent loading large arrays in memory
                if self.dataset[var].values[()] is None:
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
        return np.cumsum(np.ravel(landflag)).reshape(landflag.shape) * landflag

    def _determine_landflag(self, kv_sat):
        return (np.isfinite(kv_sat)).astype(np.int32)

    def _determine_vertical_connection(self, uzf_number):
        return uzf_number.shift(layer=-1, fill_value=0)

    def _package_data_to_sparse(self):
        notnull = self.dataset["landflag"].values == 1
        iuzno = self.dataset["iuzno"].values[notnull]
        landflag = self.dataset["landflag"].values[notnull]
        ivertcon = self.dataset["ivertcon"].values[notnull]

        ds = self.dataset[list(self._package_data)]

        layer = ds["layer"].values
        arrdict = self._ds_to_arrdict(ds)
        recarr = super().to_sparse(arrdict, layer)

        field_spec = self._get_field_spec_from_dtype(recarr)
        field_names = [i[0] for i in field_spec]
        index_spec = [("iuzno", np.int32)] + field_spec[:3]
        field_spec = (
            [("landflag", np.int32)] + [("ivertcon", np.int32)] + field_spec[3:]
        )
        sparse_dtype = np.dtype(index_spec + field_spec)

        recarr_new = np.empty(recarr.shape, dtype=sparse_dtype)
        recarr_new["iuzno"] = iuzno
        recarr_new["landflag"] = landflag
        recarr_new["ivertcon"] = ivertcon
        recarr_new[field_names] = recarr

        return recarr_new

    def render(self, directory, pkgname, globaltimes, binary):
        """Render fills in the template only, doesn't write binary data"""
        d = {}
        bin_ds = self.dataset[list(self._period_data)]
        d["periods"] = self.period_paths(
            directory, pkgname, globaltimes, bin_ds, binary=False
        )
        not_options = (
            list(self._period_data) + list(self._package_data) + ["iuzno" + "ivertcon"]
        )
        d = self.get_options(d, not_options=not_options)
        path = directory / pkgname / f"{self._pkg_id}-pkgdata.dat"
        d["packagedata"] = path.as_posix()
        d["nuzfcells"] = self._max_active_n()
        return self._template.render(d)

    def to_sparse(self, arrdict, layer):
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
        errors = super()._validate(schemata, **kwargs)

        return errors
