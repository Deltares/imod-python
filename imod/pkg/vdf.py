from imod.pkg.pkgbase import Package


class VariableDensityFlow(Package):
    _pkg_id = "vdf"
    def __init__(
        self,
        species_density,
        density_min,
        density_max,
        density_ref,
        density_concentration_slope,
        density_criterion,
        read_density,
        internodal="central",
        coupling="explicit",
        correct_water_table=False,
    ):
        super(__class__, self).__init__()
        self["species_density"] = species_density
        self["density_min"] = density_min
        self["density_max"] = density_max
        self["density_ref"] = density_ref
        self["density_concentration_slope"] = density_slope
        self["density_criterion"] = density_criterion
        self["read_density"] = read_density
        self["internodal"] = internodal
        self["coupling"] = coupling
        self["correct_water_table"] = correct_water_table
