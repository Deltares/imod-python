import textwrap

from imod.wq import VariableDensityFlow


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

    compare = textwrap.dedent(
        """\
        [vdf]
            mtdnconc = 1
            densemin = 1000.0
            densemax = 1025.0
            denseref = 1000.0
            denseslp = 0.71
            dnscrit = 0.01
            nswtcpl = 1
            iwtable = 0
            mfnadvfd = 2
        """
    )

    assert vdf._render() == compare
