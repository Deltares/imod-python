from imod.pkg import VariableDensityFlow


def test_render():
    vdf = VariableDensityFlow(
        density_species=1,
        density_min=1000.0,
        density_max=1025.0,
        density_ref=1000.0,
        density_concentration_slope=0.71,
        density_criterion=0.01,
        read_density=False,
        internodal="central",
        coupling=1,
        correct_water_table=False,
    )

    compare = (
    "[vdf]\n"
    "    mtdnconc = 1\n"
    "    densemin = 1000.0\n"
    "    densemax = 1025.0\n"
    "    denseref = 1000.0\n"
    "    denseslp = 0.71\n"
    "    dnscrit = 0.01\n"
    "    nswtcpl = 1\n"
    "    iwtable = 0\n"
    "    mfnadvfd = 2\n"
    )

    assert vdf._render() == compare
