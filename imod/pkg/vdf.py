from imod.pkg.pkgbase import Package


class VariableDensityFlow(Package):
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
        density_species,
        density_min,
        density_max,
        density_ref,
        density_concentration_slope,
        density_criterion,
        read_density,
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
