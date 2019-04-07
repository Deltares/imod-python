class VariableDensityFlow(Package):
    _pkg_id = "vdf"
    def __init__(
        self,
        species_dens,
        dens_min,
        dens_max,
        dens_ref,
        dens_slope,
        dens_criterion,
        read_dens,
        internodal="central",
        coupling="explicit",
        correct_water_table=False,
    ):
        super(__class__, self).__init__()
