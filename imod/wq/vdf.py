from imod.wq.pkgbase import Package


class VariableDensityFlow(Package):
    """
    Variable Density Flow package.

    Parameters
    ----------
    density_species: int
        is the MT3DMS species number that will be used in the equation of state
        to compute fluid density (mtdnconc).
        If density_species = 0, fluid density is specified using items 6 and 7,
        and flow will be uncoupled with transport if the IMT Process is active.
        If density_species > 0, fluid density is calculated using the MT3DMS
        species number that corresponds with density_species. A value for
        density_species greater than zero indicates that flow will be coupled
        with transport.
        If density_species = -1, fluid density is calculated using one or more
        MT3DMS species. Items 4a, 4b, and 4c will be read instead of item 4.
    density_min: float
        is the minimum fluid density (DENSEMIN). If the resulting density value
        calculated with the equation of state is less than density_min, the
        density value is set to density_min.
        If density_min = 0, the computed fluid density is not limited by
        density_min (this is the option to use for most simulations).
        If density_min > 0, a computed fluid density less than density_min is
        automatically reset to density_min.
    density_max: float
        is the maximum fluid density (DENSEMAX). If the resulting density value
        calculated with the equation of state is greater than density_max, the
        density value is set to density_max.
        If density_max = 0, the computed fluid density is not limited by
        density_max (this is the option to use for most simulations).
        If density_max > 0, a computed fluid density larger than density_max is
        automatically reset to density_max.
    density_ref: float
        is the fluid density at the reference concentration, temperature, and
        pressure (DENSEREF). For most simulations, density_ref is specified as
        the density of freshwater at 25 °C and at a reference pressure of zero.
        Value of 1000 is often used.
    density_concentration_slope: float
        is the slope d(rho)/d(C) of the linear equation of state that relates
        fluid density to solute concentration (denseslp). Value of 0.7143 is
        often used.
    density_criterion: float
        is the convergence parameter for the coupling between flow and transport
        and has units of fluid density (DNSCRIT). If the maximum density
        difference between two consecutive coupling iterations is not less than
        density_criterion, the program will continue to iterate on the flow and
        transport equations or will terminate if 'coupling' is exceeded.
    coupling: int
        is a flag used to determine the flow and transport coupling procedure
        (nswtcpl).
        If coupling = 0 or 1, flow and transport will be explicitly coupled
        using a one-timestep lag. The explicit coupling option is normally much
        faster than the iterative option and is recommended for most
        applications.
        If coupling > 1, coupling is the maximum number of non-linear coupling
        iterations for the flow and transport solutions. SEAWAT-2000 will stop
        execution after coupling iterations if convergence between flow and
        transport has not occurred.
        If coupling = -1, the flow solution will be recalculated only for: The
        first transport step of the simulation, or The last transport step of
        the MODFLOW timestep, or The maximum density change at a cell is greater
        than density_criterion.
    correct_water_table: {"False", "True"}
        is a flag used to activate the variable-density water-table corrections
        (IWTABLE).
        If correct_water_table = False, the water-table correction will not be
        applied.
        If correct_water_table = True, the water-table correction will be
        applied.
    internodal: {"upstream", "central"}
        is a flag that determines the method for calculating the internodal
        density values used to conserve fluid mass (MFNADVFD).
        If internodal = "central", internodal conductance values used to
        conserve fluid mass are calculated using a central-in-space algorithm.
        If internodal = "upstream", internodal conductance values used to
        conserve fluid mass are calculated using an upstream-weighted algorithm.
    """
    _pkg_id = "vdf"

    _template = (
        "[vdf]\n"
        "    mtdnconc = {density_species}\n"
        "    densemin = {density_min}\n"
        "    densemax = {density_max}\n"
        "    denseref = {density_ref}\n"
        "    denseslp = {density_concentration_slope}\n"
        "    dnscrit = {density_criterion}\n"
        "    nswtcpl = {coupling}\n"
        "    iwtable = {correct_water_table}\n"
        "    mfnadvfd = {internodal}\n"
    )

    _keywords = {
        "internodal": {"central": 2, "upstream": 1},
        "correct_water_table": {False: 0, True: 1},
    }

    def __init__(
        self,
        density_concentration_slope,
        density_species=1,
        density_min=1000.0,
        density_max=1025.0,
        density_ref=1000.0,
        density_criterion=0.01,
        read_density=False,
        internodal="central",
        coupling=1,
        correct_water_table=False,
    ):
        super(__class__, self).__init__()
        self["density_species"] = density_species
        self["density_min"] = density_min
        self["density_max"] = density_max
        self["density_ref"] = density_ref
        self["density_concentration_slope"] = density_concentration_slope
        self["density_criterion"] = density_criterion
        self["read_density"] = read_density
        self["internodal"] = internodal
        self["coupling"] = coupling
        self["correct_water_table"] = correct_water_table
